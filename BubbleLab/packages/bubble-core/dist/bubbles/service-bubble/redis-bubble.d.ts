import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const RedisBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"set">;
    key: z.ZodString;
    value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
    ttl: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    value: string | number | boolean | Record<string, unknown>;
    operation: "set";
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ttl?: number | undefined;
}, {
    value: string | number | boolean | Record<string, unknown>;
    operation: "set";
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ttl?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get">;
    key: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get";
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get";
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete">;
    keys: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    keys: string[];
    operation: "delete";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    keys: string[];
    operation: "delete";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"exists">;
    keys: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    keys: string[];
    operation: "exists";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    keys: string[];
    operation: "exists";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"expire">;
    key: z.ZodString;
    ttl: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "expire";
    key: string;
    ttl: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "expire";
    key: string;
    ttl: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"incr">;
    key: z.ZodString;
    amount: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "incr";
    key: string;
    amount: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "incr";
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    amount?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"decr">;
    key: z.ZodString;
    amount: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "decr";
    key: string;
    amount: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "decr";
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    amount?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"hset">;
    key: z.ZodString;
    field: z.ZodString;
    value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean]>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    value: string | number | boolean;
    operation: "hset";
    field: string;
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    value: string | number | boolean;
    operation: "hset";
    field: string;
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"hget">;
    key: z.ZodString;
    field: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "hget";
    field: string;
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "hget";
    field: string;
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"hgetall">;
    key: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "hgetall";
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "hgetall";
    key: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
type RedisBubbleParams = z.input<typeof RedisBubbleParamsSchema>;
declare const RedisBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        key: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        key?: string | undefined;
    }, {
        operation: string;
        key?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        key?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        key?: string | undefined;
    };
    data?: unknown;
}>;
type RedisBubbleResult = z.output<typeof RedisBubbleResultSchema>;
export declare class RedisBubble extends ServiceBubble<RedisBubbleParams, RedisBubbleResult> {
    static readonly service = "redis";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"set">;
        key: z.ZodString;
        value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
        ttl: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        value: string | number | boolean | Record<string, unknown>;
        operation: "set";
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ttl?: number | undefined;
    }, {
        value: string | number | boolean | Record<string, unknown>;
        operation: "set";
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ttl?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get">;
        key: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get";
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get";
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete">;
        keys: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        keys: string[];
        operation: "delete";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        keys: string[];
        operation: "delete";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"exists">;
        keys: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        keys: string[];
        operation: "exists";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        keys: string[];
        operation: "exists";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"expire">;
        key: z.ZodString;
        ttl: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "expire";
        key: string;
        ttl: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "expire";
        key: string;
        ttl: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"incr">;
        key: z.ZodString;
        amount: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "incr";
        key: string;
        amount: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "incr";
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        amount?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"decr">;
        key: z.ZodString;
        amount: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "decr";
        key: string;
        amount: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "decr";
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        amount?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"hset">;
        key: z.ZodString;
        field: z.ZodString;
        value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean]>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        value: string | number | boolean;
        operation: "hset";
        field: string;
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        value: string | number | boolean;
        operation: "hset";
        field: string;
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"hget">;
        key: z.ZodString;
        field: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "hget";
        field: string;
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "hget";
        field: string;
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"hgetall">;
        key: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "hgetall";
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "hgetall";
        key: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            key: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            key?: string | undefined;
        }, {
            operation: string;
            key?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            key?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            key?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "In-memory data store for caching and real-time applications";
    static readonly longDescription = "\n    Redis Bubble for high-performance key-value storage and caching.\n\n    Features:\n    - Ultra-fast in-memory operations\n    - Key-value pairs with optional TTL\n    - Hash data structures for complex objects\n    - Atomic increments and decrements\n    - Support for strings, numbers, booleans, and JSON\n    - Persistence options and replication\n\n    Use cases:\n    - Caching frequently accessed data\n    - Session storage\n    - Rate limiting\n    - Real-time leaderboards\n    - Pub/sub messaging\n    - Queue management\n  ";
    static readonly alias = "cache";
    private client;
    constructor(params: RedisBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getClient;
    protected performAction(context?: BubbleContext): Promise<RedisBubbleResult>;
    private set;
    private get;
    private delete;
    private exists;
    private expire;
    private incr;
    private decr;
    private hset;
    private hget;
    private hgetall;
    private extractKey;
    cleanup(): Promise<void>;
}
export {};
//# sourceMappingURL=redis-bubble.d.ts.map