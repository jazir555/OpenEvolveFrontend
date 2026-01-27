import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
export declare const ElevenLabsParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"get_signed_url">;
    agentId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_signed_url";
    agentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_signed_url";
    agentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"trigger_outbound_call">;
    agentId: z.ZodString;
    toPhoneNumber: z.ZodString;
    phoneNumberId: z.ZodOptional<z.ZodString>;
    variables: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "trigger_outbound_call";
    agentId: string;
    toPhoneNumber: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    phoneNumberId?: string | undefined;
    variables?: Record<string, string> | undefined;
}, {
    operation: "trigger_outbound_call";
    agentId: string;
    toPhoneNumber: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    phoneNumberId?: string | undefined;
    variables?: Record<string, string> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_agent">;
    agentId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_agent";
    agentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_agent";
    agentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"validate_webhook_signature">;
    signature: z.ZodString;
    timestamp: z.ZodString;
    body: z.ZodString;
    webhookSecret: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "validate_webhook_signature";
    timestamp: string;
    body: string;
    signature: string;
    webhookSecret: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "validate_webhook_signature";
    timestamp: string;
    body: string;
    signature: string;
    webhookSecret: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_conversation">;
    conversationId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_conversation";
    conversationId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_conversation";
    conversationId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_conversations">;
    agentId: z.ZodOptional<z.ZodString>;
    pageSize: z.ZodOptional<z.ZodNumber>;
    cursor: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_conversations";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cursor?: string | undefined;
    agentId?: string | undefined;
    pageSize?: number | undefined;
}, {
    operation: "get_conversations";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cursor?: string | undefined;
    agentId?: string | undefined;
    pageSize?: number | undefined;
}>]>;
export type ElevenLabsParamsInput = z.input<typeof ElevenLabsParamsSchema>;
export type ElevenLabsParamsParsed = z.output<typeof ElevenLabsParamsSchema>;
export declare const ElevenLabsResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"get_signed_url">;
    signedUrl: z.ZodOptional<z.ZodString>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_signed_url";
    signedUrl?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_signed_url";
    signedUrl?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"trigger_outbound_call">;
    callSid: z.ZodOptional<z.ZodString>;
    conversationId: z.ZodOptional<z.ZodString>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "trigger_outbound_call";
    conversationId?: string | undefined;
    callSid?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "trigger_outbound_call";
    conversationId?: string | undefined;
    callSid?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_agent">;
    agent: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_agent";
    agent?: Record<string, unknown> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_agent";
    agent?: Record<string, unknown> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"validate_webhook_signature">;
    isValid: z.ZodBoolean;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "validate_webhook_signature";
    isValid: boolean;
}, {
    error: string;
    success: boolean;
    operation: "validate_webhook_signature";
    isValid: boolean;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_conversation">;
    conversation: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_conversation";
    conversation?: Record<string, unknown> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_conversation";
    conversation?: Record<string, unknown> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_conversations">;
    conversations: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
    hasMore: z.ZodOptional<z.ZodBoolean>;
    nextCursor: z.ZodOptional<z.ZodString>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_conversations";
    conversations?: Record<string, unknown>[] | undefined;
    hasMore?: boolean | undefined;
    nextCursor?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_conversations";
    conversations?: Record<string, unknown>[] | undefined;
    hasMore?: boolean | undefined;
    nextCursor?: string | undefined;
}>]>;
export type ElevenLabsResult = z.output<typeof ElevenLabsResultSchema>;
export declare class ElevenLabsBubble extends ServiceBubble<ElevenLabsParamsParsed, ElevenLabsResult> {
    static readonly type: "service";
    static readonly service = "eleven-labs";
    static readonly authType: "apikey";
    static readonly bubbleName = "eleven-labs";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"get_signed_url">;
        agentId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_signed_url";
        agentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_signed_url";
        agentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"trigger_outbound_call">;
        agentId: z.ZodString;
        toPhoneNumber: z.ZodString;
        phoneNumberId: z.ZodOptional<z.ZodString>;
        variables: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "trigger_outbound_call";
        agentId: string;
        toPhoneNumber: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        phoneNumberId?: string | undefined;
        variables?: Record<string, string> | undefined;
    }, {
        operation: "trigger_outbound_call";
        agentId: string;
        toPhoneNumber: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        phoneNumberId?: string | undefined;
        variables?: Record<string, string> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_agent">;
        agentId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_agent";
        agentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_agent";
        agentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"validate_webhook_signature">;
        signature: z.ZodString;
        timestamp: z.ZodString;
        body: z.ZodString;
        webhookSecret: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "validate_webhook_signature";
        timestamp: string;
        body: string;
        signature: string;
        webhookSecret: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "validate_webhook_signature";
        timestamp: string;
        body: string;
        signature: string;
        webhookSecret: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_conversation">;
        conversationId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_conversation";
        conversationId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_conversation";
        conversationId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_conversations">;
        agentId: z.ZodOptional<z.ZodString>;
        pageSize: z.ZodOptional<z.ZodNumber>;
        cursor: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_conversations";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cursor?: string | undefined;
        agentId?: string | undefined;
        pageSize?: number | undefined;
    }, {
        operation: "get_conversations";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cursor?: string | undefined;
        agentId?: string | undefined;
        pageSize?: number | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"get_signed_url">;
        signedUrl: z.ZodOptional<z.ZodString>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_signed_url";
        signedUrl?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_signed_url";
        signedUrl?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"trigger_outbound_call">;
        callSid: z.ZodOptional<z.ZodString>;
        conversationId: z.ZodOptional<z.ZodString>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "trigger_outbound_call";
        conversationId?: string | undefined;
        callSid?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "trigger_outbound_call";
        conversationId?: string | undefined;
        callSid?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_agent">;
        agent: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_agent";
        agent?: Record<string, unknown> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_agent";
        agent?: Record<string, unknown> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"validate_webhook_signature">;
        isValid: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "validate_webhook_signature";
        isValid: boolean;
    }, {
        error: string;
        success: boolean;
        operation: "validate_webhook_signature";
        isValid: boolean;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_conversation">;
        conversation: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_conversation";
        conversation?: Record<string, unknown> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_conversation";
        conversation?: Record<string, unknown> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_conversations">;
        conversations: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        hasMore: z.ZodOptional<z.ZodBoolean>;
        nextCursor: z.ZodOptional<z.ZodString>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_conversations";
        conversations?: Record<string, unknown>[] | undefined;
        hasMore?: boolean | undefined;
        nextCursor?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_conversations";
        conversations?: Record<string, unknown>[] | undefined;
        hasMore?: boolean | undefined;
        nextCursor?: string | undefined;
    }>]>;
    static readonly shortDescription = "Eleven Labs integration for Conversational AI";
    static readonly longDescription = "\n    Integrate with Eleven Labs Conversational AI agents.\n    Use cases:\n    - Generate signed URLs for secure WebSocket connections to agents\n    - Trigger outbound calls\n    - Get agent details\n    - Validate webhook signatures\n    - Get conversation history\n  ";
    static readonly alias = "elevenlabs";
    constructor(params: ElevenLabsParamsInput, context?: BubbleContext);
    protected chooseCredential(): string | undefined;
    performAction(context?: BubbleContext): Promise<ElevenLabsResult>;
    testCredential(): Promise<boolean>;
    private getSignedUrl;
    private triggerOutboundCall;
    private getAgent;
    private validateWebhookSignature;
    private getConversation;
    private getConversations;
}
//# sourceMappingURL=eleven-labs.d.ts.map