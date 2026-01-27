import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const SlackBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"sendMessage">;
    channel: z.ZodString;
    text: z.ZodString;
    threadTs: z.ZodOptional<z.ZodString>;
    blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "sendMessage";
    channel: string;
    text: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    blocks?: unknown[] | undefined;
    threadTs?: string | undefined;
}, {
    operation: "sendMessage";
    channel: string;
    text: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    blocks?: unknown[] | undefined;
    threadTs?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"sendDM">;
    userId: z.ZodString;
    text: z.ZodString;
    blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "sendDM";
    text: string;
    userId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    blocks?: unknown[] | undefined;
}, {
    operation: "sendDM";
    text: string;
    userId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    blocks?: unknown[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateMessage">;
    channel: z.ZodString;
    timestamp: z.ZodString;
    text: z.ZodString;
    blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateMessage";
    channel: string;
    text: string;
    timestamp: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    blocks?: unknown[] | undefined;
}, {
    operation: "updateMessage";
    channel: string;
    text: string;
    timestamp: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    blocks?: unknown[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteMessage">;
    channel: z.ZodString;
    timestamp: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteMessage";
    channel: string;
    timestamp: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteMessage";
    channel: string;
    timestamp: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"addReaction">;
    channel: z.ZodString;
    timestamp: z.ZodString;
    reaction: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "addReaction";
    channel: string;
    timestamp: string;
    reaction: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "addReaction";
    channel: string;
    timestamp: string;
    reaction: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"removeReaction">;
    channel: z.ZodString;
    timestamp: z.ZodString;
    reaction: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "removeReaction";
    channel: string;
    timestamp: string;
    reaction: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "removeReaction";
    channel: string;
    timestamp: string;
    reaction: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getChannelInfo">;
    channelId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getChannelInfo";
    channelId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getChannelInfo";
    channelId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listChannels">;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    types: z.ZodOptional<z.ZodArray<z.ZodEnum<["public_channel", "private_channel", "mpim", "im"]>, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listChannels";
    limit: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    types?: ("public_channel" | "private_channel" | "mpim" | "im")[] | undefined;
}, {
    operation: "listChannels";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    types?: ("public_channel" | "private_channel" | "mpim" | "im")[] | undefined;
    limit?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getUserInfo">;
    userId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getUserInfo";
    userId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getUserInfo";
    userId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"uploadFile">;
    channel: z.ZodString;
    fileContent: z.ZodUnion<[z.ZodString, z.ZodType<Buffer<ArrayBufferLike>, z.ZodTypeDef, Buffer<ArrayBufferLike>>]>;
    filename: z.ZodString;
    filetype: z.ZodOptional<z.ZodString>;
    title: z.ZodOptional<z.ZodString>;
    initialComment: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "uploadFile";
    channel: string;
    filename: string;
    fileContent: string | Buffer<ArrayBufferLike>;
    title?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    filetype?: string | undefined;
    initialComment?: string | undefined;
}, {
    operation: "uploadFile";
    channel: string;
    filename: string;
    fileContent: string | Buffer<ArrayBufferLike>;
    title?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    filetype?: string | undefined;
    initialComment?: string | undefined;
}>]>;
type SlackBubbleParams = z.input<typeof SlackBubbleParamsSchema>;
declare const SlackBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        channel: z.ZodOptional<z.ZodString>;
        timestamp: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        channel?: string | undefined;
        timestamp?: string | undefined;
    }, {
        operation: string;
        channel?: string | undefined;
        timestamp?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        channel?: string | undefined;
        timestamp?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        channel?: string | undefined;
        timestamp?: string | undefined;
    };
    data?: unknown;
}>;
type SlackBubbleResult = z.output<typeof SlackBubbleResultSchema>;
export declare class SlackBubble extends ServiceBubble<SlackBubbleParams, SlackBubbleResult> {
    static readonly service = "slack";
    static readonly authType: "oauth";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"sendMessage">;
        channel: z.ZodString;
        text: z.ZodString;
        threadTs: z.ZodOptional<z.ZodString>;
        blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "sendMessage";
        channel: string;
        text: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        blocks?: unknown[] | undefined;
        threadTs?: string | undefined;
    }, {
        operation: "sendMessage";
        channel: string;
        text: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        blocks?: unknown[] | undefined;
        threadTs?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"sendDM">;
        userId: z.ZodString;
        text: z.ZodString;
        blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "sendDM";
        text: string;
        userId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        blocks?: unknown[] | undefined;
    }, {
        operation: "sendDM";
        text: string;
        userId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        blocks?: unknown[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateMessage">;
        channel: z.ZodString;
        timestamp: z.ZodString;
        text: z.ZodString;
        blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateMessage";
        channel: string;
        text: string;
        timestamp: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        blocks?: unknown[] | undefined;
    }, {
        operation: "updateMessage";
        channel: string;
        text: string;
        timestamp: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        blocks?: unknown[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteMessage">;
        channel: z.ZodString;
        timestamp: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteMessage";
        channel: string;
        timestamp: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteMessage";
        channel: string;
        timestamp: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"addReaction">;
        channel: z.ZodString;
        timestamp: z.ZodString;
        reaction: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "addReaction";
        channel: string;
        timestamp: string;
        reaction: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "addReaction";
        channel: string;
        timestamp: string;
        reaction: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"removeReaction">;
        channel: z.ZodString;
        timestamp: z.ZodString;
        reaction: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "removeReaction";
        channel: string;
        timestamp: string;
        reaction: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "removeReaction";
        channel: string;
        timestamp: string;
        reaction: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getChannelInfo">;
        channelId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getChannelInfo";
        channelId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getChannelInfo";
        channelId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listChannels">;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        types: z.ZodOptional<z.ZodArray<z.ZodEnum<["public_channel", "private_channel", "mpim", "im"]>, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listChannels";
        limit: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        types?: ("public_channel" | "private_channel" | "mpim" | "im")[] | undefined;
    }, {
        operation: "listChannels";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        types?: ("public_channel" | "private_channel" | "mpim" | "im")[] | undefined;
        limit?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getUserInfo">;
        userId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getUserInfo";
        userId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getUserInfo";
        userId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"uploadFile">;
        channel: z.ZodString;
        fileContent: z.ZodUnion<[z.ZodString, z.ZodType<Buffer<ArrayBufferLike>, z.ZodTypeDef, Buffer<ArrayBufferLike>>]>;
        filename: z.ZodString;
        filetype: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodString>;
        initialComment: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "uploadFile";
        channel: string;
        filename: string;
        fileContent: string | Buffer<ArrayBufferLike>;
        title?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        filetype?: string | undefined;
        initialComment?: string | undefined;
    }, {
        operation: "uploadFile";
        channel: string;
        filename: string;
        fileContent: string | Buffer<ArrayBufferLike>;
        title?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        filetype?: string | undefined;
        initialComment?: string | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            channel: z.ZodOptional<z.ZodString>;
            timestamp: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            channel?: string | undefined;
            timestamp?: string | undefined;
        }, {
            operation: string;
            channel?: string | undefined;
            timestamp?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            channel?: string | undefined;
            timestamp?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            channel?: string | undefined;
            timestamp?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Team communication and collaboration platform";
    static readonly longDescription = "\n    Slack Bubble for team communication and messaging.\n\n    Features:\n    - Send messages to channels and direct messages\n    - Rich formatting with Block Kit\n    - Threaded conversations\n    - File sharing and uploads\n    - Message reactions\n    - Channel and user information retrieval\n    - Real-time webhooks support\n\n    Use cases:\n    - Team notifications and alerts\n    - Automated status updates\n    - Incident management\n    - Approval workflows\n    - Daily standups and reports\n    - Integration notifications\n  ";
    static readonly alias = "chat";
    private botToken;
    private baseUrl;
    constructor(params: SlackBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getToken;
    protected performAction(context?: BubbleContext): Promise<SlackBubbleResult>;
    private makeRequest;
    private sendMessage;
    private sendDM;
    private updateMessage;
    private deleteMessage;
    private addReaction;
    private removeReaction;
    private getChannelInfo;
    private listChannels;
    private getUserInfo;
    private uploadFile;
    private extractChannel;
}
export {};
//# sourceMappingURL=slack-bubble.d.ts.map