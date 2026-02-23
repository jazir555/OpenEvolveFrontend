/**
 * Agentic Context Engine Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Agentic Context Engine
 * interactions. All adapters must normalize their data to/from this format.
 */
import { z } from 'zod';
/**
 * Context Type Enum
 */
export declare const ContextType: z.ZodEnum<["conversation", "document", "session", "user_profile", "domain_knowledge", "working_memory", "episodic"]>;
export type ContextType = z.infer<typeof ContextType>;
/**
 * Context State Enum
 */
export declare const ContextState: z.ZodEnum<["active", "archived", "expired", "locked"]>;
export type ContextState = z.infer<typeof ContextState>;
/**
 * Context Entry Schema
 */
export declare const ContextEntry: z.ZodObject<{
    key: z.ZodString;
    value: z.ZodAny;
    type: z.ZodOptional<z.ZodEnum<["string", "number", "boolean", "object", "array", "null"]>>;
    timestamp: z.ZodOptional<z.ZodString>;
    ttl_seconds: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    key: string;
    timestamp?: string | undefined;
    metadata?: Record<string, any> | undefined;
    type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
    value?: any;
    ttl_seconds?: number | undefined;
}, {
    key: string;
    timestamp?: string | undefined;
    metadata?: Record<string, any> | undefined;
    type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
    value?: any;
    ttl_seconds?: number | undefined;
}>;
export type ContextEntry = z.infer<typeof ContextEntry>;
/**
 * Context Store Request Schema
 */
export declare const AgenticContextRequest: z.ZodObject<{
    context_id: z.ZodOptional<z.ZodString>;
    context_type: z.ZodEnum<["conversation", "document", "session", "user_profile", "domain_knowledge", "working_memory", "episodic"]>;
    entries: z.ZodOptional<z.ZodArray<z.ZodObject<{
        key: z.ZodString;
        value: z.ZodAny;
        type: z.ZodOptional<z.ZodEnum<["string", "number", "boolean", "object", "array", "null"]>>;
        timestamp: z.ZodOptional<z.ZodString>;
        ttl_seconds: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }, {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }>, "many">>;
    state: z.ZodOptional<z.ZodEnum<["active", "archived", "expired", "locked"]>>;
    config: z.ZodOptional<z.ZodObject<{
        ttl_seconds: z.ZodOptional<z.ZodNumber>;
        max_entries: z.ZodOptional<z.ZodNumber>;
        eviction_policy: z.ZodOptional<z.ZodEnum<["lru", "fifo", "lfu"]>>;
        compression: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        compression?: boolean | undefined;
        ttl_seconds?: number | undefined;
        max_entries?: number | undefined;
        eviction_policy?: "lru" | "fifo" | "lfu" | undefined;
    }, {
        compression?: boolean | undefined;
        ttl_seconds?: number | undefined;
        max_entries?: number | undefined;
        eviction_policy?: "lru" | "fifo" | "lfu" | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    context_type: "document" | "session" | "conversation" | "user_profile" | "domain_knowledge" | "working_memory" | "episodic";
    correlation_id?: string | undefined;
    state?: "active" | "archived" | "expired" | "locked" | undefined;
    config?: {
        compression?: boolean | undefined;
        ttl_seconds?: number | undefined;
        max_entries?: number | undefined;
        eviction_policy?: "lru" | "fifo" | "lfu" | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    entries?: {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }[] | undefined;
    context_id?: string | undefined;
}, {
    timeout_ms: number;
    context_type: "document" | "session" | "conversation" | "user_profile" | "domain_knowledge" | "working_memory" | "episodic";
    correlation_id?: string | undefined;
    state?: "active" | "archived" | "expired" | "locked" | undefined;
    config?: {
        compression?: boolean | undefined;
        ttl_seconds?: number | undefined;
        max_entries?: number | undefined;
        eviction_policy?: "lru" | "fifo" | "lfu" | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    entries?: {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }[] | undefined;
    context_id?: string | undefined;
}>;
export type AgenticContextRequest = z.infer<typeof AgenticContextRequest>;
/**
 * Context Store Response Schema
 */
export declare const AgenticContextResponse: z.ZodObject<{
    context_id: z.ZodString;
    context_type: z.ZodEnum<["conversation", "document", "session", "user_profile", "domain_knowledge", "working_memory", "episodic"]>;
    state: z.ZodEnum<["active", "archived", "expired", "locked"]>;
    entries: z.ZodArray<z.ZodObject<{
        key: z.ZodString;
        value: z.ZodAny;
        type: z.ZodOptional<z.ZodEnum<["string", "number", "boolean", "object", "array", "null"]>>;
        timestamp: z.ZodOptional<z.ZodString>;
        ttl_seconds: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }, {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }>, "many">;
    metadata: z.ZodOptional<z.ZodObject<{
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        expires_at: z.ZodOptional<z.ZodString>;
        entry_count: z.ZodOptional<z.ZodNumber>;
        size_bytes: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        size_bytes?: number | undefined;
        expires_at?: string | undefined;
        entry_count?: number | undefined;
    }, {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        size_bytes?: number | undefined;
        expires_at?: string | undefined;
        entry_count?: number | undefined;
    }>>;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    state: "active" | "archived" | "expired" | "locked";
    entries: {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }[];
    context_id: string;
    context_type: "document" | "session" | "conversation" | "user_profile" | "domain_knowledge" | "working_memory" | "episodic";
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        size_bytes?: number | undefined;
        expires_at?: string | undefined;
        entry_count?: number | undefined;
    } | undefined;
}, {
    timestamp: string;
    state: "active" | "archived" | "expired" | "locked";
    entries: {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }[];
    context_id: string;
    context_type: "document" | "session" | "conversation" | "user_profile" | "domain_knowledge" | "working_memory" | "episodic";
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        size_bytes?: number | undefined;
        expires_at?: string | undefined;
        entry_count?: number | undefined;
    } | undefined;
}>;
export type AgenticContextResponse = z.infer<typeof AgenticContextResponse>;
/**
 * Context Query Request Schema
 */
export declare const AgenticContextQueryRequest: z.ZodObject<{
    context_id: z.ZodString;
    query: z.ZodObject<{
        keys: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        pattern: z.ZodOptional<z.ZodString>;
        filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        filter?: Record<string, any> | undefined;
        keys?: string[] | undefined;
        limit?: number | undefined;
        pattern?: string | undefined;
        offset?: number | undefined;
    }, {
        filter?: Record<string, any> | undefined;
        keys?: string[] | undefined;
        limit?: number | undefined;
        pattern?: string | undefined;
        offset?: number | undefined;
    }>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    query: {
        filter?: Record<string, any> | undefined;
        keys?: string[] | undefined;
        limit?: number | undefined;
        pattern?: string | undefined;
        offset?: number | undefined;
    };
    context_id: string;
    correlation_id?: string | undefined;
}, {
    timeout_ms: number;
    query: {
        filter?: Record<string, any> | undefined;
        keys?: string[] | undefined;
        limit?: number | undefined;
        pattern?: string | undefined;
        offset?: number | undefined;
    };
    context_id: string;
    correlation_id?: string | undefined;
}>;
export type AgenticContextQueryRequest = z.infer<typeof AgenticContextQueryRequest>;
/**
 * Context Query Response Schema
 */
export declare const AgenticContextQueryResponse: z.ZodObject<{
    context_id: z.ZodString;
    results: z.ZodArray<z.ZodObject<{
        key: z.ZodString;
        value: z.ZodAny;
        type: z.ZodOptional<z.ZodEnum<["string", "number", "boolean", "object", "array", "null"]>>;
        timestamp: z.ZodOptional<z.ZodString>;
        ttl_seconds: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }, {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }>, "many">;
    metadata: z.ZodOptional<z.ZodObject<{
        total_count: z.ZodOptional<z.ZodNumber>;
        returned_count: z.ZodNumber;
        query_time_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        returned_count: number;
        total_count?: number | undefined;
        query_time_ms?: number | undefined;
    }, {
        returned_count: number;
        total_count?: number | undefined;
        query_time_ms?: number | undefined;
    }>>;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
    }, {
        message: string;
        code: string;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    results: {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }[];
    context_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
    } | undefined;
    metadata?: {
        returned_count: number;
        total_count?: number | undefined;
        query_time_ms?: number | undefined;
    } | undefined;
}, {
    timestamp: string;
    results: {
        key: string;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
        value?: any;
        ttl_seconds?: number | undefined;
    }[];
    context_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
    } | undefined;
    metadata?: {
        returned_count: number;
        total_count?: number | undefined;
        query_time_ms?: number | undefined;
    } | undefined;
}>;
export type AgenticContextQueryResponse = z.infer<typeof AgenticContextQueryResponse>;
/**
 * Context Update Request Schema
 */
export declare const AgenticContextUpdateRequest: z.ZodObject<{
    context_id: z.ZodString;
    updates: z.ZodObject<{
        add_entries: z.ZodOptional<z.ZodArray<z.ZodObject<{
            key: z.ZodString;
            value: z.ZodAny;
            type: z.ZodOptional<z.ZodEnum<["string", "number", "boolean", "object", "array", "null"]>>;
            timestamp: z.ZodOptional<z.ZodString>;
            ttl_seconds: z.ZodOptional<z.ZodNumber>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            key: string;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
            value?: any;
            ttl_seconds?: number | undefined;
        }, {
            key: string;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
            value?: any;
            ttl_seconds?: number | undefined;
        }>, "many">>;
        update_entries: z.ZodOptional<z.ZodArray<z.ZodObject<{
            key: z.ZodString;
            value: z.ZodAny;
            merge: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            key: string;
            value?: any;
            merge?: boolean | undefined;
        }, {
            key: string;
            value?: any;
            merge?: boolean | undefined;
        }>, "many">>;
        delete_keys: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        set_state: z.ZodOptional<z.ZodEnum<["active", "archived", "expired", "locked"]>>;
    }, "strip", z.ZodTypeAny, {
        add_entries?: {
            key: string;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
            value?: any;
            ttl_seconds?: number | undefined;
        }[] | undefined;
        update_entries?: {
            key: string;
            value?: any;
            merge?: boolean | undefined;
        }[] | undefined;
        delete_keys?: string[] | undefined;
        set_state?: "active" | "archived" | "expired" | "locked" | undefined;
    }, {
        add_entries?: {
            key: string;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
            value?: any;
            ttl_seconds?: number | undefined;
        }[] | undefined;
        update_entries?: {
            key: string;
            value?: any;
            merge?: boolean | undefined;
        }[] | undefined;
        delete_keys?: string[] | undefined;
        set_state?: "active" | "archived" | "expired" | "locked" | undefined;
    }>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    context_id: string;
    updates: {
        add_entries?: {
            key: string;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
            value?: any;
            ttl_seconds?: number | undefined;
        }[] | undefined;
        update_entries?: {
            key: string;
            value?: any;
            merge?: boolean | undefined;
        }[] | undefined;
        delete_keys?: string[] | undefined;
        set_state?: "active" | "archived" | "expired" | "locked" | undefined;
    };
    correlation_id?: string | undefined;
}, {
    timeout_ms: number;
    context_id: string;
    updates: {
        add_entries?: {
            key: string;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            type?: "string" | "number" | "boolean" | "object" | "null" | "array" | undefined;
            value?: any;
            ttl_seconds?: number | undefined;
        }[] | undefined;
        update_entries?: {
            key: string;
            value?: any;
            merge?: boolean | undefined;
        }[] | undefined;
        delete_keys?: string[] | undefined;
        set_state?: "active" | "archived" | "expired" | "locked" | undefined;
    };
    correlation_id?: string | undefined;
}>;
export type AgenticContextUpdateRequest = z.infer<typeof AgenticContextUpdateRequest>;
/**
 * Error Model
 */
export declare const AgenticContextError: z.ZodObject<{
    code: z.ZodEnum<["CONTEXT_NOT_FOUND", "CONTEXT_EXPIRED", "CONTEXT_LOCKED", "INVALID_KEY", "QUOTA_EXCEEDED", "VALIDATION_ERROR", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "QUOTA_EXCEEDED" | "CONTEXT_NOT_FOUND" | "CONTEXT_EXPIRED" | "CONTEXT_LOCKED" | "INVALID_KEY";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "QUOTA_EXCEEDED" | "CONTEXT_NOT_FOUND" | "CONTEXT_EXPIRED" | "CONTEXT_LOCKED" | "INVALID_KEY";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type AgenticContextError = z.infer<typeof AgenticContextError>;
/**
 * Validation Functions
 */
export declare function validateAgenticContextRequest(data: unknown): {
    success: boolean;
    data?: AgenticContextRequest;
    errors?: string[];
};
export declare function isAgenticContextRequest(data: unknown): data is AgenticContextRequest;
/**
 * Examples
 */
export declare const AgenticContextExamples: {
    validRequest: AgenticContextRequest;
    validResponse: AgenticContextResponse;
};
//# sourceMappingURL=agentic-context-engine-canonical.d.ts.map