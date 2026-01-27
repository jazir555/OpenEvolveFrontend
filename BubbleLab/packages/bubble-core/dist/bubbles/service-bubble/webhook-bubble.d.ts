import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const WebhookBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"receiveWebhook">;
    path: z.ZodString;
    headers: z.ZodRecord<z.ZodString, z.ZodString>;
    body: z.ZodAny;
    signature: z.ZodOptional<z.ZodString>;
    signatureAlgorithm: z.ZodDefault<z.ZodOptional<z.ZodEnum<["hmac-sha1", "hmac-sha256", "aws-v4"]>>>;
    secret: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodOptional<z.ZodString>;
    maxAge: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    store: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    contentType: z.ZodOptional<z.ZodString>;
    maxPayloadSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    path: string;
    operation: "receiveWebhook";
    headers: Record<string, string>;
    maxAge: number;
    store: boolean;
    signatureAlgorithm: "hmac-sha1" | "hmac-sha256" | "aws-v4";
    maxPayloadSize: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    timestamp?: string | undefined;
    body?: any;
    contentType?: string | undefined;
    signature?: string | undefined;
    secret?: string | undefined;
}, {
    path: string;
    operation: "receiveWebhook";
    headers: Record<string, string>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    timestamp?: string | undefined;
    body?: any;
    contentType?: string | undefined;
    signature?: string | undefined;
    maxAge?: number | undefined;
    store?: boolean | undefined;
    secret?: string | undefined;
    signatureAlgorithm?: "hmac-sha1" | "hmac-sha256" | "aws-v4" | undefined;
    maxPayloadSize?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"verifySignature">;
    payload: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
    signature: z.ZodString;
    secret: z.ZodString;
    algorithm: z.ZodDefault<z.ZodOptional<z.ZodEnum<["hmac-sha1", "hmac-sha256", "aws-v4"]>>>;
    provider: z.ZodDefault<z.ZodOptional<z.ZodEnum<["github", "stripe", "slack", "twilio", "generic"]>>>;
    timestamp: z.ZodOptional<z.ZodString>;
    maxAge: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    provider: "slack" | "github" | "stripe" | "twilio" | "generic";
    operation: "verifySignature";
    payload: string | Record<string, unknown>;
    signature: string;
    maxAge: number;
    secret: string;
    algorithm: "hmac-sha1" | "hmac-sha256" | "aws-v4";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    timestamp?: string | undefined;
}, {
    operation: "verifySignature";
    payload: string | Record<string, unknown>;
    signature: string;
    secret: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    provider?: "slack" | "github" | "stripe" | "twilio" | "generic" | undefined;
    timestamp?: string | undefined;
    maxAge?: number | undefined;
    algorithm?: "hmac-sha1" | "hmac-sha256" | "aws-v4" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"parsePayload">;
    provider: z.ZodEnum<["github", "gitlab", "bitbucket", "slack", "stripe", "shopify", "paypal", "generic"]>;
    payload: z.ZodAny;
    headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    provider: "slack" | "github" | "stripe" | "generic" | "gitlab" | "bitbucket" | "shopify" | "paypal";
    operation: "parsePayload";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    headers?: Record<string, string> | undefined;
    payload?: any;
}, {
    provider: "slack" | "github" | "stripe" | "generic" | "gitlab" | "bitbucket" | "shopify" | "paypal";
    operation: "parsePayload";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    headers?: Record<string, string> | undefined;
    payload?: any;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"validateSignature">;
    payload: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
    signature: z.ZodString;
    secret: z.ZodString;
    algorithm: z.ZodDefault<z.ZodOptional<z.ZodEnum<["hmac-sha1", "hmac-sha256", "aws-v4"]>>>;
    signatureHeader: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    timestamp: z.ZodOptional<z.ZodString>;
    maxAge: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "validateSignature";
    payload: string | Record<string, unknown>;
    signature: string;
    maxAge: number;
    secret: string;
    algorithm: "hmac-sha1" | "hmac-sha256" | "aws-v4";
    signatureHeader: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    timestamp?: string | undefined;
}, {
    operation: "validateSignature";
    payload: string | Record<string, unknown>;
    signature: string;
    secret: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    timestamp?: string | undefined;
    maxAge?: number | undefined;
    algorithm?: "hmac-sha1" | "hmac-sha256" | "aws-v4" | undefined;
    signatureHeader?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"dispatchEvent">;
    eventType: z.ZodString;
    payload: z.ZodAny;
    targets: z.ZodArray<z.ZodString, "many">;
    headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    retries: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    operation: "dispatchEvent";
    retries: number;
    eventType: string;
    targets: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    headers?: Record<string, string> | undefined;
    payload?: any;
}, {
    operation: "dispatchEvent";
    eventType: string;
    targets: string[];
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    headers?: Record<string, string> | undefined;
    payload?: any;
    retries?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"registerHandler">;
    eventType: z.ZodString;
    handlerUrl: z.ZodString;
    filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    retries: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    operation: "registerHandler";
    retries: number;
    eventType: string;
    handlerUrl: string;
    filter?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "registerHandler";
    eventType: string;
    handlerUrl: string;
    timeout?: number | undefined;
    filter?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    retries?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"unregisterHandler">;
    handlerId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "unregisterHandler";
    handlerId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "unregisterHandler";
    handlerId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"retryFailedWebhook">;
    webhookId: z.ZodString;
    retryCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    maxRetries: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    backoffMs: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    maxRetries: number;
    operation: "retryFailedWebhook";
    retryCount: number;
    webhookId: string;
    backoffMs: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "retryFailedWebhook";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxRetries?: number | undefined;
    retryCount?: number | undefined;
    backoffMs?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRetryStatus">;
    webhookId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getRetryStatus";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getRetryStatus";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listWebhooks">;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    filter: z.ZodOptional<z.ZodObject<{
        path: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodString>;
        startDate: z.ZodOptional<z.ZodString>;
        endDate: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        path?: string | undefined;
        provider?: string | undefined;
        startDate?: string | undefined;
        endDate?: string | undefined;
    }, {
        path?: string | undefined;
        provider?: string | undefined;
        startDate?: string | undefined;
        endDate?: string | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listWebhooks";
    limit: number;
    offset: number;
    filter?: {
        path?: string | undefined;
        provider?: string | undefined;
        startDate?: string | undefined;
        endDate?: string | undefined;
    } | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "listWebhooks";
    filter?: {
        path?: string | undefined;
        provider?: string | undefined;
        startDate?: string | undefined;
        endDate?: string | undefined;
    } | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getWebhook">;
    webhookId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getWebhook";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getWebhook";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"replayWebhook">;
    webhookId: z.ZodString;
    targets: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "replayWebhook";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    targets?: string[] | undefined;
}, {
    operation: "replayWebhook";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    targets?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteWebhook">;
    webhookId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteWebhook";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteWebhook";
    webhookId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getStats">;
    webhookId: z.ZodOptional<z.ZodString>;
    path: z.ZodOptional<z.ZodString>;
    timeRange: z.ZodDefault<z.ZodOptional<z.ZodEnum<["hour", "day", "week", "month"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getStats";
    timeRange: "hour" | "week" | "month" | "day";
    path?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    webhookId?: string | undefined;
}, {
    operation: "getStats";
    path?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    webhookId?: string | undefined;
    timeRange?: "hour" | "week" | "month" | "day" | undefined;
}>]>;
type WebhookBubbleParams = z.input<typeof WebhookBubbleParamsSchema>;
declare const WebhookBubbleResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"receiveWebhook">;
    result: z.ZodObject<{
        webhookId: z.ZodString;
        receivedAt: z.ZodString;
        path: z.ZodString;
        provider: z.ZodOptional<z.ZodString>;
        eventType: z.ZodOptional<z.ZodString>;
        validated: z.ZodBoolean;
        parsed: z.ZodBoolean;
        stored: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        path: string;
        success: boolean;
        webhookId: string;
        receivedAt: string;
        validated: boolean;
        parsed: boolean;
        stored: boolean;
        provider?: string | undefined;
        eventType?: string | undefined;
    }, {
        error: string;
        path: string;
        success: boolean;
        webhookId: string;
        receivedAt: string;
        validated: boolean;
        parsed: boolean;
        stored: boolean;
        provider?: string | undefined;
        eventType?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "receiveWebhook";
    result: {
        error: string;
        path: string;
        success: boolean;
        webhookId: string;
        receivedAt: string;
        validated: boolean;
        parsed: boolean;
        stored: boolean;
        provider?: string | undefined;
        eventType?: string | undefined;
    };
}, {
    operation: "receiveWebhook";
    result: {
        error: string;
        path: string;
        success: boolean;
        webhookId: string;
        receivedAt: string;
        validated: boolean;
        parsed: boolean;
        stored: boolean;
        provider?: string | undefined;
        eventType?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"verifySignature">;
    result: z.ZodObject<{
        valid: z.ZodBoolean;
        algorithm: z.ZodString;
        provider: z.ZodString;
        timestampValid: z.ZodOptional<z.ZodBoolean>;
        expectedSignature: z.ZodOptional<z.ZodString>;
        receivedSignature: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        valid: boolean;
        success: boolean;
        provider: string;
        algorithm: string;
        receivedSignature: string;
        expectedSignature?: string | undefined;
        timestampValid?: boolean | undefined;
    }, {
        error: string;
        valid: boolean;
        success: boolean;
        provider: string;
        algorithm: string;
        receivedSignature: string;
        expectedSignature?: string | undefined;
        timestampValid?: boolean | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "verifySignature";
    result: {
        error: string;
        valid: boolean;
        success: boolean;
        provider: string;
        algorithm: string;
        receivedSignature: string;
        expectedSignature?: string | undefined;
        timestampValid?: boolean | undefined;
    };
}, {
    operation: "verifySignature";
    result: {
        error: string;
        valid: boolean;
        success: boolean;
        provider: string;
        algorithm: string;
        receivedSignature: string;
        expectedSignature?: string | undefined;
        timestampValid?: boolean | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"parsePayload">;
    result: z.ZodObject<{
        provider: z.ZodString;
        eventType: z.ZodString;
        data: z.ZodOptional<z.ZodAny>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        parsedAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        provider: string;
        eventType: string;
        parsedAt: string;
        data?: any;
        metadata?: Record<string, unknown> | undefined;
    }, {
        error: string;
        success: boolean;
        provider: string;
        eventType: string;
        parsedAt: string;
        data?: any;
        metadata?: Record<string, unknown> | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "parsePayload";
    result: {
        error: string;
        success: boolean;
        provider: string;
        eventType: string;
        parsedAt: string;
        data?: any;
        metadata?: Record<string, unknown> | undefined;
    };
}, {
    operation: "parsePayload";
    result: {
        error: string;
        success: boolean;
        provider: string;
        eventType: string;
        parsedAt: string;
        data?: any;
        metadata?: Record<string, unknown> | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"validateSignature">;
    result: z.ZodObject<{
        valid: z.ZodBoolean;
        algorithm: z.ZodString;
        expectedSignature: z.ZodString;
        receivedSignature: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        valid: boolean;
        success: boolean;
        algorithm: string;
        expectedSignature: string;
        receivedSignature: string;
    }, {
        error: string;
        valid: boolean;
        success: boolean;
        algorithm: string;
        expectedSignature: string;
        receivedSignature: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "validateSignature";
    result: {
        error: string;
        valid: boolean;
        success: boolean;
        algorithm: string;
        expectedSignature: string;
        receivedSignature: string;
    };
}, {
    operation: "validateSignature";
    result: {
        error: string;
        valid: boolean;
        success: boolean;
        algorithm: string;
        expectedSignature: string;
        receivedSignature: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"dispatchEvent">;
    result: z.ZodObject<{
        eventId: z.ZodString;
        dispatchedAt: z.ZodString;
        targets: z.ZodArray<z.ZodObject<{
            url: z.ZodString;
            status: z.ZodEnum<["pending", "success", "failed"]>;
            statusCode: z.ZodOptional<z.ZodNumber>;
            responseTime: z.ZodOptional<z.ZodNumber>;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }, {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }>, "many">;
        totalTargets: z.ZodNumber;
        successfulTargets: z.ZodNumber;
        failedTargets: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        targets: {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }[];
        eventId: string;
        dispatchedAt: string;
        totalTargets: number;
        successfulTargets: number;
        failedTargets: number;
    }, {
        error: string;
        success: boolean;
        targets: {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }[];
        eventId: string;
        dispatchedAt: string;
        totalTargets: number;
        successfulTargets: number;
        failedTargets: number;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "dispatchEvent";
    result: {
        error: string;
        success: boolean;
        targets: {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }[];
        eventId: string;
        dispatchedAt: string;
        totalTargets: number;
        successfulTargets: number;
        failedTargets: number;
    };
}, {
    operation: "dispatchEvent";
    result: {
        error: string;
        success: boolean;
        targets: {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }[];
        eventId: string;
        dispatchedAt: string;
        totalTargets: number;
        successfulTargets: number;
        failedTargets: number;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"registerHandler">;
    result: z.ZodObject<{
        handlerId: z.ZodString;
        eventType: z.ZodString;
        handlerUrl: z.ZodString;
        registeredAt: z.ZodString;
        active: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        active: boolean;
        eventType: string;
        handlerUrl: string;
        handlerId: string;
        registeredAt: string;
    }, {
        error: string;
        success: boolean;
        active: boolean;
        eventType: string;
        handlerUrl: string;
        handlerId: string;
        registeredAt: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "registerHandler";
    result: {
        error: string;
        success: boolean;
        active: boolean;
        eventType: string;
        handlerUrl: string;
        handlerId: string;
        registeredAt: string;
    };
}, {
    operation: "registerHandler";
    result: {
        error: string;
        success: boolean;
        active: boolean;
        eventType: string;
        handlerUrl: string;
        handlerId: string;
        registeredAt: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"unregisterHandler">;
    result: z.ZodObject<{
        handlerId: z.ZodString;
        unregistered: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        handlerId: string;
        unregistered: boolean;
    }, {
        error: string;
        success: boolean;
        handlerId: string;
        unregistered: boolean;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "unregisterHandler";
    result: {
        error: string;
        success: boolean;
        handlerId: string;
        unregistered: boolean;
    };
}, {
    operation: "unregisterHandler";
    result: {
        error: string;
        success: boolean;
        handlerId: string;
        unregistered: boolean;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"retryFailedWebhook">;
    result: z.ZodObject<{
        webhookId: z.ZodString;
        retryAttempt: z.ZodNumber;
        maxRetries: z.ZodNumber;
        status: z.ZodEnum<["pending", "success", "failed", "exhausted"]>;
        nextRetryAt: z.ZodOptional<z.ZodString>;
        retryHistory: z.ZodArray<z.ZodObject<{
            attempt: z.ZodNumber;
            timestamp: z.ZodString;
            status: z.ZodString;
            responseTime: z.ZodOptional<z.ZodNumber>;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }, {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }>, "many">;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: "success" | "failed" | "pending" | "exhausted";
        success: boolean;
        maxRetries: number;
        webhookId: string;
        retryAttempt: number;
        retryHistory: {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }[];
        nextRetryAt?: string | undefined;
    }, {
        error: string;
        status: "success" | "failed" | "pending" | "exhausted";
        success: boolean;
        maxRetries: number;
        webhookId: string;
        retryAttempt: number;
        retryHistory: {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }[];
        nextRetryAt?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "retryFailedWebhook";
    result: {
        error: string;
        status: "success" | "failed" | "pending" | "exhausted";
        success: boolean;
        maxRetries: number;
        webhookId: string;
        retryAttempt: number;
        retryHistory: {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }[];
        nextRetryAt?: string | undefined;
    };
}, {
    operation: "retryFailedWebhook";
    result: {
        error: string;
        status: "success" | "failed" | "pending" | "exhausted";
        success: boolean;
        maxRetries: number;
        webhookId: string;
        retryAttempt: number;
        retryHistory: {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }[];
        nextRetryAt?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRetryStatus">;
    result: z.ZodObject<{
        webhookId: z.ZodString;
        retryCount: z.ZodNumber;
        maxRetries: z.ZodNumber;
        status: z.ZodEnum<["pending", "success", "failed", "exhausted"]>;
        retryHistory: z.ZodArray<z.ZodObject<{
            attempt: z.ZodNumber;
            timestamp: z.ZodString;
            status: z.ZodString;
            responseTime: z.ZodOptional<z.ZodNumber>;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }, {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }>, "many">;
        nextRetryAt: z.ZodOptional<z.ZodString>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: "success" | "failed" | "pending" | "exhausted";
        success: boolean;
        maxRetries: number;
        retryCount: number;
        webhookId: string;
        retryHistory: {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }[];
        nextRetryAt?: string | undefined;
    }, {
        error: string;
        status: "success" | "failed" | "pending" | "exhausted";
        success: boolean;
        maxRetries: number;
        retryCount: number;
        webhookId: string;
        retryHistory: {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }[];
        nextRetryAt?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getRetryStatus";
    result: {
        error: string;
        status: "success" | "failed" | "pending" | "exhausted";
        success: boolean;
        maxRetries: number;
        retryCount: number;
        webhookId: string;
        retryHistory: {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }[];
        nextRetryAt?: string | undefined;
    };
}, {
    operation: "getRetryStatus";
    result: {
        error: string;
        status: "success" | "failed" | "pending" | "exhausted";
        success: boolean;
        maxRetries: number;
        retryCount: number;
        webhookId: string;
        retryHistory: {
            status: string;
            timestamp: string;
            attempt: number;
            error?: string | undefined;
            responseTime?: number | undefined;
        }[];
        nextRetryAt?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listWebhooks">;
    result: z.ZodObject<{
        webhooks: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            receivedAt: z.ZodString;
            path: z.ZodString;
            provider: z.ZodOptional<z.ZodString>;
            eventType: z.ZodOptional<z.ZodString>;
            validated: z.ZodBoolean;
            processed: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            path: string;
            id: string;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        }, {
            path: string;
            id: string;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        }>, "many">;
        count: z.ZodNumber;
        limit: z.ZodNumber;
        offset: z.ZodNumber;
        total: z.ZodOptional<z.ZodNumber>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        webhooks: {
            path: string;
            id: string;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        }[];
        total?: number | undefined;
    }, {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        webhooks: {
            path: string;
            id: string;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        }[];
        total?: number | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "listWebhooks";
    result: {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        webhooks: {
            path: string;
            id: string;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        }[];
        total?: number | undefined;
    };
}, {
    operation: "listWebhooks";
    result: {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        webhooks: {
            path: string;
            id: string;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        }[];
        total?: number | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getWebhook">;
    result: z.ZodObject<{
        webhook: z.ZodObject<{
            id: z.ZodString;
            receivedAt: z.ZodString;
            path: z.ZodString;
            headers: z.ZodRecord<z.ZodString, z.ZodString>;
            body: z.ZodAny;
            provider: z.ZodOptional<z.ZodString>;
            eventType: z.ZodOptional<z.ZodString>;
            validated: z.ZodBoolean;
            parsed: z.ZodBoolean;
            processed: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            path: string;
            id: string;
            headers: Record<string, string>;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            provider?: string | undefined;
            body?: any;
            eventType?: string | undefined;
        }, {
            path: string;
            id: string;
            headers: Record<string, string>;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            provider?: string | undefined;
            body?: any;
            eventType?: string | undefined;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        webhook: {
            path: string;
            id: string;
            headers: Record<string, string>;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            provider?: string | undefined;
            body?: any;
            eventType?: string | undefined;
        };
        success: boolean;
    }, {
        error: string;
        webhook: {
            path: string;
            id: string;
            headers: Record<string, string>;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            provider?: string | undefined;
            body?: any;
            eventType?: string | undefined;
        };
        success: boolean;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getWebhook";
    result: {
        error: string;
        webhook: {
            path: string;
            id: string;
            headers: Record<string, string>;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            provider?: string | undefined;
            body?: any;
            eventType?: string | undefined;
        };
        success: boolean;
    };
}, {
    operation: "getWebhook";
    result: {
        error: string;
        webhook: {
            path: string;
            id: string;
            headers: Record<string, string>;
            processed: boolean;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            provider?: string | undefined;
            body?: any;
            eventType?: string | undefined;
        };
        success: boolean;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"replayWebhook">;
    result: z.ZodObject<{
        webhookId: z.ZodString;
        replayedAt: z.ZodString;
        originalReceivedAt: z.ZodString;
        targets: z.ZodArray<z.ZodObject<{
            url: z.ZodString;
            status: z.ZodEnum<["pending", "success", "failed"]>;
            statusCode: z.ZodOptional<z.ZodNumber>;
            responseTime: z.ZodOptional<z.ZodNumber>;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }, {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }>, "many">;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        targets: {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }[];
        webhookId: string;
        replayedAt: string;
        originalReceivedAt: string;
    }, {
        error: string;
        success: boolean;
        targets: {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }[];
        webhookId: string;
        replayedAt: string;
        originalReceivedAt: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "replayWebhook";
    result: {
        error: string;
        success: boolean;
        targets: {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }[];
        webhookId: string;
        replayedAt: string;
        originalReceivedAt: string;
    };
}, {
    operation: "replayWebhook";
    result: {
        error: string;
        success: boolean;
        targets: {
            status: "success" | "failed" | "pending";
            url: string;
            error?: string | undefined;
            responseTime?: number | undefined;
            statusCode?: number | undefined;
        }[];
        webhookId: string;
        replayedAt: string;
        originalReceivedAt: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteWebhook">;
    result: z.ZodObject<{
        deleted: z.ZodBoolean;
        webhookId: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        deleted: boolean;
        webhookId: string;
    }, {
        error: string;
        success: boolean;
        deleted: boolean;
        webhookId: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteWebhook";
    result: {
        error: string;
        success: boolean;
        deleted: boolean;
        webhookId: string;
    };
}, {
    operation: "deleteWebhook";
    result: {
        error: string;
        success: boolean;
        deleted: boolean;
        webhookId: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getStats">;
    result: z.ZodObject<{
        webhookId: z.ZodOptional<z.ZodString>;
        path: z.ZodOptional<z.ZodString>;
        timeRange: z.ZodString;
        metrics: z.ZodObject<{
            totalReceived: z.ZodNumber;
            totalValidated: z.ZodNumber;
            totalParsed: z.ZodNumber;
            totalDispatched: z.ZodNumber;
            validationFailureRate: z.ZodNumber;
            averageProcessingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            totalReceived: number;
            totalValidated: number;
            totalParsed: number;
            totalDispatched: number;
            validationFailureRate: number;
            averageProcessingTime: number;
        }, {
            totalReceived: number;
            totalValidated: number;
            totalParsed: number;
            totalDispatched: number;
            validationFailureRate: number;
            averageProcessingTime: number;
        }>;
        topEventTypes: z.ZodArray<z.ZodObject<{
            eventType: z.ZodString;
            count: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            count: number;
            eventType: string;
        }, {
            count: number;
            eventType: string;
        }>, "many">;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        metrics: {
            totalReceived: number;
            totalValidated: number;
            totalParsed: number;
            totalDispatched: number;
            validationFailureRate: number;
            averageProcessingTime: number;
        };
        timeRange: string;
        topEventTypes: {
            count: number;
            eventType: string;
        }[];
        path?: string | undefined;
        webhookId?: string | undefined;
    }, {
        error: string;
        success: boolean;
        metrics: {
            totalReceived: number;
            totalValidated: number;
            totalParsed: number;
            totalDispatched: number;
            validationFailureRate: number;
            averageProcessingTime: number;
        };
        timeRange: string;
        topEventTypes: {
            count: number;
            eventType: string;
        }[];
        path?: string | undefined;
        webhookId?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getStats";
    result: {
        error: string;
        success: boolean;
        metrics: {
            totalReceived: number;
            totalValidated: number;
            totalParsed: number;
            totalDispatched: number;
            validationFailureRate: number;
            averageProcessingTime: number;
        };
        timeRange: string;
        topEventTypes: {
            count: number;
            eventType: string;
        }[];
        path?: string | undefined;
        webhookId?: string | undefined;
    };
}, {
    operation: "getStats";
    result: {
        error: string;
        success: boolean;
        metrics: {
            totalReceived: number;
            totalValidated: number;
            totalParsed: number;
            totalDispatched: number;
            validationFailureRate: number;
            averageProcessingTime: number;
        };
        timeRange: string;
        topEventTypes: {
            count: number;
            eventType: string;
        }[];
        path?: string | undefined;
        webhookId?: string | undefined;
    };
}>]>;
type WebhookBubbleResult = z.output<typeof WebhookBubbleResultSchema>;
export declare class WebhookBubble<T extends WebhookBubbleParams = WebhookBubbleParams> extends ServiceBubble<T, any> {
    static readonly type: "service";
    static readonly service = "webhook";
    static readonly authType: "none";
    static readonly bubbleName = "webhook";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"receiveWebhook">;
        path: z.ZodString;
        headers: z.ZodRecord<z.ZodString, z.ZodString>;
        body: z.ZodAny;
        signature: z.ZodOptional<z.ZodString>;
        signatureAlgorithm: z.ZodDefault<z.ZodOptional<z.ZodEnum<["hmac-sha1", "hmac-sha256", "aws-v4"]>>>;
        secret: z.ZodOptional<z.ZodString>;
        timestamp: z.ZodOptional<z.ZodString>;
        maxAge: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        store: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        contentType: z.ZodOptional<z.ZodString>;
        maxPayloadSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        operation: "receiveWebhook";
        headers: Record<string, string>;
        maxAge: number;
        store: boolean;
        signatureAlgorithm: "hmac-sha1" | "hmac-sha256" | "aws-v4";
        maxPayloadSize: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        timestamp?: string | undefined;
        body?: any;
        contentType?: string | undefined;
        signature?: string | undefined;
        secret?: string | undefined;
    }, {
        path: string;
        operation: "receiveWebhook";
        headers: Record<string, string>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        timestamp?: string | undefined;
        body?: any;
        contentType?: string | undefined;
        signature?: string | undefined;
        maxAge?: number | undefined;
        store?: boolean | undefined;
        secret?: string | undefined;
        signatureAlgorithm?: "hmac-sha1" | "hmac-sha256" | "aws-v4" | undefined;
        maxPayloadSize?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"verifySignature">;
        payload: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
        signature: z.ZodString;
        secret: z.ZodString;
        algorithm: z.ZodDefault<z.ZodOptional<z.ZodEnum<["hmac-sha1", "hmac-sha256", "aws-v4"]>>>;
        provider: z.ZodDefault<z.ZodOptional<z.ZodEnum<["github", "stripe", "slack", "twilio", "generic"]>>>;
        timestamp: z.ZodOptional<z.ZodString>;
        maxAge: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        provider: "slack" | "github" | "stripe" | "twilio" | "generic";
        operation: "verifySignature";
        payload: string | Record<string, unknown>;
        signature: string;
        maxAge: number;
        secret: string;
        algorithm: "hmac-sha1" | "hmac-sha256" | "aws-v4";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        timestamp?: string | undefined;
    }, {
        operation: "verifySignature";
        payload: string | Record<string, unknown>;
        signature: string;
        secret: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        provider?: "slack" | "github" | "stripe" | "twilio" | "generic" | undefined;
        timestamp?: string | undefined;
        maxAge?: number | undefined;
        algorithm?: "hmac-sha1" | "hmac-sha256" | "aws-v4" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"parsePayload">;
        provider: z.ZodEnum<["github", "gitlab", "bitbucket", "slack", "stripe", "shopify", "paypal", "generic"]>;
        payload: z.ZodAny;
        headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        provider: "slack" | "github" | "stripe" | "generic" | "gitlab" | "bitbucket" | "shopify" | "paypal";
        operation: "parsePayload";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        headers?: Record<string, string> | undefined;
        payload?: any;
    }, {
        provider: "slack" | "github" | "stripe" | "generic" | "gitlab" | "bitbucket" | "shopify" | "paypal";
        operation: "parsePayload";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        headers?: Record<string, string> | undefined;
        payload?: any;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"validateSignature">;
        payload: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
        signature: z.ZodString;
        secret: z.ZodString;
        algorithm: z.ZodDefault<z.ZodOptional<z.ZodEnum<["hmac-sha1", "hmac-sha256", "aws-v4"]>>>;
        signatureHeader: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        timestamp: z.ZodOptional<z.ZodString>;
        maxAge: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "validateSignature";
        payload: string | Record<string, unknown>;
        signature: string;
        maxAge: number;
        secret: string;
        algorithm: "hmac-sha1" | "hmac-sha256" | "aws-v4";
        signatureHeader: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        timestamp?: string | undefined;
    }, {
        operation: "validateSignature";
        payload: string | Record<string, unknown>;
        signature: string;
        secret: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        timestamp?: string | undefined;
        maxAge?: number | undefined;
        algorithm?: "hmac-sha1" | "hmac-sha256" | "aws-v4" | undefined;
        signatureHeader?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"dispatchEvent">;
        eventType: z.ZodString;
        payload: z.ZodAny;
        targets: z.ZodArray<z.ZodString, "many">;
        headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        retries: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        operation: "dispatchEvent";
        retries: number;
        eventType: string;
        targets: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        headers?: Record<string, string> | undefined;
        payload?: any;
    }, {
        operation: "dispatchEvent";
        eventType: string;
        targets: string[];
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        headers?: Record<string, string> | undefined;
        payload?: any;
        retries?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"registerHandler">;
        eventType: z.ZodString;
        handlerUrl: z.ZodString;
        filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        retries: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        operation: "registerHandler";
        retries: number;
        eventType: string;
        handlerUrl: string;
        filter?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "registerHandler";
        eventType: string;
        handlerUrl: string;
        timeout?: number | undefined;
        filter?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        retries?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"unregisterHandler">;
        handlerId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "unregisterHandler";
        handlerId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "unregisterHandler";
        handlerId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"retryFailedWebhook">;
        webhookId: z.ZodString;
        retryCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        maxRetries: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        backoffMs: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        maxRetries: number;
        operation: "retryFailedWebhook";
        retryCount: number;
        webhookId: string;
        backoffMs: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "retryFailedWebhook";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxRetries?: number | undefined;
        retryCount?: number | undefined;
        backoffMs?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRetryStatus">;
        webhookId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRetryStatus";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getRetryStatus";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listWebhooks">;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        filter: z.ZodOptional<z.ZodObject<{
            path: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodString>;
            startDate: z.ZodOptional<z.ZodString>;
            endDate: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            path?: string | undefined;
            provider?: string | undefined;
            startDate?: string | undefined;
            endDate?: string | undefined;
        }, {
            path?: string | undefined;
            provider?: string | undefined;
            startDate?: string | undefined;
            endDate?: string | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listWebhooks";
        limit: number;
        offset: number;
        filter?: {
            path?: string | undefined;
            provider?: string | undefined;
            startDate?: string | undefined;
            endDate?: string | undefined;
        } | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "listWebhooks";
        filter?: {
            path?: string | undefined;
            provider?: string | undefined;
            startDate?: string | undefined;
            endDate?: string | undefined;
        } | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getWebhook">;
        webhookId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getWebhook";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getWebhook";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"replayWebhook">;
        webhookId: z.ZodString;
        targets: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "replayWebhook";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        targets?: string[] | undefined;
    }, {
        operation: "replayWebhook";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        targets?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteWebhook">;
        webhookId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteWebhook";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteWebhook";
        webhookId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getStats">;
        webhookId: z.ZodOptional<z.ZodString>;
        path: z.ZodOptional<z.ZodString>;
        timeRange: z.ZodDefault<z.ZodOptional<z.ZodEnum<["hour", "day", "week", "month"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getStats";
        timeRange: "hour" | "week" | "month" | "day";
        path?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        webhookId?: string | undefined;
    }, {
        operation: "getStats";
        path?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        webhookId?: string | undefined;
        timeRange?: "hour" | "week" | "month" | "day" | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"receiveWebhook">;
        result: z.ZodObject<{
            webhookId: z.ZodString;
            receivedAt: z.ZodString;
            path: z.ZodString;
            provider: z.ZodOptional<z.ZodString>;
            eventType: z.ZodOptional<z.ZodString>;
            validated: z.ZodBoolean;
            parsed: z.ZodBoolean;
            stored: z.ZodBoolean;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            path: string;
            success: boolean;
            webhookId: string;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            stored: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        }, {
            error: string;
            path: string;
            success: boolean;
            webhookId: string;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            stored: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "receiveWebhook";
        result: {
            error: string;
            path: string;
            success: boolean;
            webhookId: string;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            stored: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        };
    }, {
        operation: "receiveWebhook";
        result: {
            error: string;
            path: string;
            success: boolean;
            webhookId: string;
            receivedAt: string;
            validated: boolean;
            parsed: boolean;
            stored: boolean;
            provider?: string | undefined;
            eventType?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"verifySignature">;
        result: z.ZodObject<{
            valid: z.ZodBoolean;
            algorithm: z.ZodString;
            provider: z.ZodString;
            timestampValid: z.ZodOptional<z.ZodBoolean>;
            expectedSignature: z.ZodOptional<z.ZodString>;
            receivedSignature: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            valid: boolean;
            success: boolean;
            provider: string;
            algorithm: string;
            receivedSignature: string;
            expectedSignature?: string | undefined;
            timestampValid?: boolean | undefined;
        }, {
            error: string;
            valid: boolean;
            success: boolean;
            provider: string;
            algorithm: string;
            receivedSignature: string;
            expectedSignature?: string | undefined;
            timestampValid?: boolean | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "verifySignature";
        result: {
            error: string;
            valid: boolean;
            success: boolean;
            provider: string;
            algorithm: string;
            receivedSignature: string;
            expectedSignature?: string | undefined;
            timestampValid?: boolean | undefined;
        };
    }, {
        operation: "verifySignature";
        result: {
            error: string;
            valid: boolean;
            success: boolean;
            provider: string;
            algorithm: string;
            receivedSignature: string;
            expectedSignature?: string | undefined;
            timestampValid?: boolean | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"parsePayload">;
        result: z.ZodObject<{
            provider: z.ZodString;
            eventType: z.ZodString;
            data: z.ZodOptional<z.ZodAny>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            parsedAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            provider: string;
            eventType: string;
            parsedAt: string;
            data?: any;
            metadata?: Record<string, unknown> | undefined;
        }, {
            error: string;
            success: boolean;
            provider: string;
            eventType: string;
            parsedAt: string;
            data?: any;
            metadata?: Record<string, unknown> | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "parsePayload";
        result: {
            error: string;
            success: boolean;
            provider: string;
            eventType: string;
            parsedAt: string;
            data?: any;
            metadata?: Record<string, unknown> | undefined;
        };
    }, {
        operation: "parsePayload";
        result: {
            error: string;
            success: boolean;
            provider: string;
            eventType: string;
            parsedAt: string;
            data?: any;
            metadata?: Record<string, unknown> | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"validateSignature">;
        result: z.ZodObject<{
            valid: z.ZodBoolean;
            algorithm: z.ZodString;
            expectedSignature: z.ZodString;
            receivedSignature: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            valid: boolean;
            success: boolean;
            algorithm: string;
            expectedSignature: string;
            receivedSignature: string;
        }, {
            error: string;
            valid: boolean;
            success: boolean;
            algorithm: string;
            expectedSignature: string;
            receivedSignature: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "validateSignature";
        result: {
            error: string;
            valid: boolean;
            success: boolean;
            algorithm: string;
            expectedSignature: string;
            receivedSignature: string;
        };
    }, {
        operation: "validateSignature";
        result: {
            error: string;
            valid: boolean;
            success: boolean;
            algorithm: string;
            expectedSignature: string;
            receivedSignature: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"dispatchEvent">;
        result: z.ZodObject<{
            eventId: z.ZodString;
            dispatchedAt: z.ZodString;
            targets: z.ZodArray<z.ZodObject<{
                url: z.ZodString;
                status: z.ZodEnum<["pending", "success", "failed"]>;
                statusCode: z.ZodOptional<z.ZodNumber>;
                responseTime: z.ZodOptional<z.ZodNumber>;
                error: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }, {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }>, "many">;
            totalTargets: z.ZodNumber;
            successfulTargets: z.ZodNumber;
            failedTargets: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            targets: {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }[];
            eventId: string;
            dispatchedAt: string;
            totalTargets: number;
            successfulTargets: number;
            failedTargets: number;
        }, {
            error: string;
            success: boolean;
            targets: {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }[];
            eventId: string;
            dispatchedAt: string;
            totalTargets: number;
            successfulTargets: number;
            failedTargets: number;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "dispatchEvent";
        result: {
            error: string;
            success: boolean;
            targets: {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }[];
            eventId: string;
            dispatchedAt: string;
            totalTargets: number;
            successfulTargets: number;
            failedTargets: number;
        };
    }, {
        operation: "dispatchEvent";
        result: {
            error: string;
            success: boolean;
            targets: {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }[];
            eventId: string;
            dispatchedAt: string;
            totalTargets: number;
            successfulTargets: number;
            failedTargets: number;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"registerHandler">;
        result: z.ZodObject<{
            handlerId: z.ZodString;
            eventType: z.ZodString;
            handlerUrl: z.ZodString;
            registeredAt: z.ZodString;
            active: z.ZodBoolean;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            active: boolean;
            eventType: string;
            handlerUrl: string;
            handlerId: string;
            registeredAt: string;
        }, {
            error: string;
            success: boolean;
            active: boolean;
            eventType: string;
            handlerUrl: string;
            handlerId: string;
            registeredAt: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "registerHandler";
        result: {
            error: string;
            success: boolean;
            active: boolean;
            eventType: string;
            handlerUrl: string;
            handlerId: string;
            registeredAt: string;
        };
    }, {
        operation: "registerHandler";
        result: {
            error: string;
            success: boolean;
            active: boolean;
            eventType: string;
            handlerUrl: string;
            handlerId: string;
            registeredAt: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"unregisterHandler">;
        result: z.ZodObject<{
            handlerId: z.ZodString;
            unregistered: z.ZodBoolean;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            handlerId: string;
            unregistered: boolean;
        }, {
            error: string;
            success: boolean;
            handlerId: string;
            unregistered: boolean;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "unregisterHandler";
        result: {
            error: string;
            success: boolean;
            handlerId: string;
            unregistered: boolean;
        };
    }, {
        operation: "unregisterHandler";
        result: {
            error: string;
            success: boolean;
            handlerId: string;
            unregistered: boolean;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"retryFailedWebhook">;
        result: z.ZodObject<{
            webhookId: z.ZodString;
            retryAttempt: z.ZodNumber;
            maxRetries: z.ZodNumber;
            status: z.ZodEnum<["pending", "success", "failed", "exhausted"]>;
            nextRetryAt: z.ZodOptional<z.ZodString>;
            retryHistory: z.ZodArray<z.ZodObject<{
                attempt: z.ZodNumber;
                timestamp: z.ZodString;
                status: z.ZodString;
                responseTime: z.ZodOptional<z.ZodNumber>;
                error: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }, {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }>, "many">;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: "success" | "failed" | "pending" | "exhausted";
            success: boolean;
            maxRetries: number;
            webhookId: string;
            retryAttempt: number;
            retryHistory: {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }[];
            nextRetryAt?: string | undefined;
        }, {
            error: string;
            status: "success" | "failed" | "pending" | "exhausted";
            success: boolean;
            maxRetries: number;
            webhookId: string;
            retryAttempt: number;
            retryHistory: {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }[];
            nextRetryAt?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "retryFailedWebhook";
        result: {
            error: string;
            status: "success" | "failed" | "pending" | "exhausted";
            success: boolean;
            maxRetries: number;
            webhookId: string;
            retryAttempt: number;
            retryHistory: {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }[];
            nextRetryAt?: string | undefined;
        };
    }, {
        operation: "retryFailedWebhook";
        result: {
            error: string;
            status: "success" | "failed" | "pending" | "exhausted";
            success: boolean;
            maxRetries: number;
            webhookId: string;
            retryAttempt: number;
            retryHistory: {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }[];
            nextRetryAt?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRetryStatus">;
        result: z.ZodObject<{
            webhookId: z.ZodString;
            retryCount: z.ZodNumber;
            maxRetries: z.ZodNumber;
            status: z.ZodEnum<["pending", "success", "failed", "exhausted"]>;
            retryHistory: z.ZodArray<z.ZodObject<{
                attempt: z.ZodNumber;
                timestamp: z.ZodString;
                status: z.ZodString;
                responseTime: z.ZodOptional<z.ZodNumber>;
                error: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }, {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }>, "many">;
            nextRetryAt: z.ZodOptional<z.ZodString>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: "success" | "failed" | "pending" | "exhausted";
            success: boolean;
            maxRetries: number;
            retryCount: number;
            webhookId: string;
            retryHistory: {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }[];
            nextRetryAt?: string | undefined;
        }, {
            error: string;
            status: "success" | "failed" | "pending" | "exhausted";
            success: boolean;
            maxRetries: number;
            retryCount: number;
            webhookId: string;
            retryHistory: {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }[];
            nextRetryAt?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRetryStatus";
        result: {
            error: string;
            status: "success" | "failed" | "pending" | "exhausted";
            success: boolean;
            maxRetries: number;
            retryCount: number;
            webhookId: string;
            retryHistory: {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }[];
            nextRetryAt?: string | undefined;
        };
    }, {
        operation: "getRetryStatus";
        result: {
            error: string;
            status: "success" | "failed" | "pending" | "exhausted";
            success: boolean;
            maxRetries: number;
            retryCount: number;
            webhookId: string;
            retryHistory: {
                status: string;
                timestamp: string;
                attempt: number;
                error?: string | undefined;
                responseTime?: number | undefined;
            }[];
            nextRetryAt?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listWebhooks">;
        result: z.ZodObject<{
            webhooks: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                receivedAt: z.ZodString;
                path: z.ZodString;
                provider: z.ZodOptional<z.ZodString>;
                eventType: z.ZodOptional<z.ZodString>;
                validated: z.ZodBoolean;
                processed: z.ZodBoolean;
            }, "strip", z.ZodTypeAny, {
                path: string;
                id: string;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                provider?: string | undefined;
                eventType?: string | undefined;
            }, {
                path: string;
                id: string;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                provider?: string | undefined;
                eventType?: string | undefined;
            }>, "many">;
            count: z.ZodNumber;
            limit: z.ZodNumber;
            offset: z.ZodNumber;
            total: z.ZodOptional<z.ZodNumber>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            webhooks: {
                path: string;
                id: string;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                provider?: string | undefined;
                eventType?: string | undefined;
            }[];
            total?: number | undefined;
        }, {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            webhooks: {
                path: string;
                id: string;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                provider?: string | undefined;
                eventType?: string | undefined;
            }[];
            total?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "listWebhooks";
        result: {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            webhooks: {
                path: string;
                id: string;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                provider?: string | undefined;
                eventType?: string | undefined;
            }[];
            total?: number | undefined;
        };
    }, {
        operation: "listWebhooks";
        result: {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            webhooks: {
                path: string;
                id: string;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                provider?: string | undefined;
                eventType?: string | undefined;
            }[];
            total?: number | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getWebhook">;
        result: z.ZodObject<{
            webhook: z.ZodObject<{
                id: z.ZodString;
                receivedAt: z.ZodString;
                path: z.ZodString;
                headers: z.ZodRecord<z.ZodString, z.ZodString>;
                body: z.ZodAny;
                provider: z.ZodOptional<z.ZodString>;
                eventType: z.ZodOptional<z.ZodString>;
                validated: z.ZodBoolean;
                parsed: z.ZodBoolean;
                processed: z.ZodBoolean;
            }, "strip", z.ZodTypeAny, {
                path: string;
                id: string;
                headers: Record<string, string>;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                parsed: boolean;
                provider?: string | undefined;
                body?: any;
                eventType?: string | undefined;
            }, {
                path: string;
                id: string;
                headers: Record<string, string>;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                parsed: boolean;
                provider?: string | undefined;
                body?: any;
                eventType?: string | undefined;
            }>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            webhook: {
                path: string;
                id: string;
                headers: Record<string, string>;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                parsed: boolean;
                provider?: string | undefined;
                body?: any;
                eventType?: string | undefined;
            };
            success: boolean;
        }, {
            error: string;
            webhook: {
                path: string;
                id: string;
                headers: Record<string, string>;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                parsed: boolean;
                provider?: string | undefined;
                body?: any;
                eventType?: string | undefined;
            };
            success: boolean;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getWebhook";
        result: {
            error: string;
            webhook: {
                path: string;
                id: string;
                headers: Record<string, string>;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                parsed: boolean;
                provider?: string | undefined;
                body?: any;
                eventType?: string | undefined;
            };
            success: boolean;
        };
    }, {
        operation: "getWebhook";
        result: {
            error: string;
            webhook: {
                path: string;
                id: string;
                headers: Record<string, string>;
                processed: boolean;
                receivedAt: string;
                validated: boolean;
                parsed: boolean;
                provider?: string | undefined;
                body?: any;
                eventType?: string | undefined;
            };
            success: boolean;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"replayWebhook">;
        result: z.ZodObject<{
            webhookId: z.ZodString;
            replayedAt: z.ZodString;
            originalReceivedAt: z.ZodString;
            targets: z.ZodArray<z.ZodObject<{
                url: z.ZodString;
                status: z.ZodEnum<["pending", "success", "failed"]>;
                statusCode: z.ZodOptional<z.ZodNumber>;
                responseTime: z.ZodOptional<z.ZodNumber>;
                error: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }, {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }>, "many">;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            targets: {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }[];
            webhookId: string;
            replayedAt: string;
            originalReceivedAt: string;
        }, {
            error: string;
            success: boolean;
            targets: {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }[];
            webhookId: string;
            replayedAt: string;
            originalReceivedAt: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "replayWebhook";
        result: {
            error: string;
            success: boolean;
            targets: {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }[];
            webhookId: string;
            replayedAt: string;
            originalReceivedAt: string;
        };
    }, {
        operation: "replayWebhook";
        result: {
            error: string;
            success: boolean;
            targets: {
                status: "success" | "failed" | "pending";
                url: string;
                error?: string | undefined;
                responseTime?: number | undefined;
                statusCode?: number | undefined;
            }[];
            webhookId: string;
            replayedAt: string;
            originalReceivedAt: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteWebhook">;
        result: z.ZodObject<{
            deleted: z.ZodBoolean;
            webhookId: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            deleted: boolean;
            webhookId: string;
        }, {
            error: string;
            success: boolean;
            deleted: boolean;
            webhookId: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteWebhook";
        result: {
            error: string;
            success: boolean;
            deleted: boolean;
            webhookId: string;
        };
    }, {
        operation: "deleteWebhook";
        result: {
            error: string;
            success: boolean;
            deleted: boolean;
            webhookId: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getStats">;
        result: z.ZodObject<{
            webhookId: z.ZodOptional<z.ZodString>;
            path: z.ZodOptional<z.ZodString>;
            timeRange: z.ZodString;
            metrics: z.ZodObject<{
                totalReceived: z.ZodNumber;
                totalValidated: z.ZodNumber;
                totalParsed: z.ZodNumber;
                totalDispatched: z.ZodNumber;
                validationFailureRate: z.ZodNumber;
                averageProcessingTime: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                totalReceived: number;
                totalValidated: number;
                totalParsed: number;
                totalDispatched: number;
                validationFailureRate: number;
                averageProcessingTime: number;
            }, {
                totalReceived: number;
                totalValidated: number;
                totalParsed: number;
                totalDispatched: number;
                validationFailureRate: number;
                averageProcessingTime: number;
            }>;
            topEventTypes: z.ZodArray<z.ZodObject<{
                eventType: z.ZodString;
                count: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                count: number;
                eventType: string;
            }, {
                count: number;
                eventType: string;
            }>, "many">;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            metrics: {
                totalReceived: number;
                totalValidated: number;
                totalParsed: number;
                totalDispatched: number;
                validationFailureRate: number;
                averageProcessingTime: number;
            };
            timeRange: string;
            topEventTypes: {
                count: number;
                eventType: string;
            }[];
            path?: string | undefined;
            webhookId?: string | undefined;
        }, {
            error: string;
            success: boolean;
            metrics: {
                totalReceived: number;
                totalValidated: number;
                totalParsed: number;
                totalDispatched: number;
                validationFailureRate: number;
                averageProcessingTime: number;
            };
            timeRange: string;
            topEventTypes: {
                count: number;
                eventType: string;
            }[];
            path?: string | undefined;
            webhookId?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getStats";
        result: {
            error: string;
            success: boolean;
            metrics: {
                totalReceived: number;
                totalValidated: number;
                totalParsed: number;
                totalDispatched: number;
                validationFailureRate: number;
                averageProcessingTime: number;
            };
            timeRange: string;
            topEventTypes: {
                count: number;
                eventType: string;
            }[];
            path?: string | undefined;
            webhookId?: string | undefined;
        };
    }, {
        operation: "getStats";
        result: {
            error: string;
            success: boolean;
            metrics: {
                totalReceived: number;
                totalValidated: number;
                totalParsed: number;
                totalDispatched: number;
                validationFailureRate: number;
                averageProcessingTime: number;
            };
            timeRange: string;
            topEventTypes: {
                count: number;
                eventType: string;
            }[];
            path?: string | undefined;
            webhookId?: string | undefined;
        };
    }>]>;
    static readonly shortDescription = "Complete webhook management and processing service";
    static readonly longDescription = "\n    Comprehensive webhook service for receiving, parsing, validating, and dispatching webhooks.\n\n    Operations (12 Total):\n    1. receiveWebhook - Receive and validate incoming webhook requests\n    2. verifySignature - Verify webhook signature with multiple algorithms\n    3. parsePayload - Parse webhook payloads from different providers\n    4. validateSignature - Legacy signature validation method\n    5. dispatchEvent - Dispatch webhook events to multiple targets\n    6. registerHandler - Register webhook event handlers\n    7. unregisterHandler - Unregister webhook event handlers\n    8. retryFailedWebhook - Retry failed webhooks with exponential backoff\n    9. getRetryStatus - Get retry status and history\n    10. listWebhooks - List stored webhooks with filtering\n    11. getWebhook - Get webhook details by ID\n    12. replayWebhook - Replay previously received webhooks\n    13. deleteWebhook - Delete stored webhooks\n    14. getStats - Get webhook statistics and metrics\n\n    Supported Providers:\n    - GitHub\n    - GitLab\n    - Bitbucket\n    - Slack\n    - Stripe\n    - Shopify\n    - PayPal\n    - Twilio\n    - Generic (custom webhooks)\n\n    Features:\n    - Signature validation (HMAC-SHA1, HMAC-SHA256, AWS Signature V4)\n    - Timestamp validation (replay attack prevention)\n    - IP whitelist validation (optional)\n    - Rate limiting (receive: 100/min, dispatch: 50/min)\n    - Payload size limits (max 10MB per webhook)\n    - Content-Type validation\n    - Handler registration and management\n    - Automatic retry with exponential backoff (1m, 5m, 15m, 30m, 1h)\n    - Webhook storage and replay\n    - Statistics and metrics\n    - Full resilience patterns\n  ";
    static readonly alias = "webhook";
    private static storage;
    private resilience;
    constructor(params: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    protected chooseCredential(): string | undefined;
    protected performAction(context?: BubbleContext): Promise<Extract<WebhookBubbleResult, {
        operation: T['operation'];
    }>>;
    private receiveWebhook;
    private parsePayload;
    private validateSignature;
    private dispatchEvent;
    private replayWebhook;
    private listWebhooks;
    private deleteWebhook;
    private getStats;
    private verifySignature;
    private registerHandler;
    private unregisterHandler;
    private retryFailedWebhook;
    private getRetryStatus;
    private getWebhook;
    private detectProvider;
    private parsePayloadInternal;
    private validateSignatureInternal;
    private generateId;
}
export {};
//# sourceMappingURL=webhook-bubble.d.ts.map