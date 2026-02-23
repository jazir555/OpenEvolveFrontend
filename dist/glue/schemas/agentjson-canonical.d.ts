/**
 * AgentJSON Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for AgentJSON
 * (JSON-based agent protocol and serialization) interactions.
 */
import { z } from 'zod';
/**
 * Agent Type Enum
 */
export declare const AgentType: z.ZodEnum<["reactive", "proactive", "deliberative", "hybrid"]>;
export type AgentType = z.infer<typeof AgentType>;
/**
 * Message Type Enum
 */
export declare const MessageType: z.ZodEnum<["request", "response", "notification", "error", "heartbeat"]>;
export type MessageType = z.infer<typeof MessageType>;
/**
 * Agent State Enum
 */
export declare const AgentState: z.ZodEnum<["idle", "processing", "waiting", "error", "terminated"]>;
export type AgentState = z.infer<typeof AgentState>;
/**
 * Agent Message Schema
 */
export declare const AgentMessage: z.ZodObject<{
    message_id: z.ZodString;
    message_type: z.ZodEnum<["request", "response", "notification", "error", "heartbeat"]>;
    sender_id: z.ZodString;
    receiver_id: z.ZodOptional<z.ZodString>;
    payload: z.ZodRecord<z.ZodString, z.ZodAny>;
    timestamp: z.ZodString;
    correlation_id: z.ZodOptional<z.ZodString>;
    reply_to: z.ZodOptional<z.ZodString>;
    expires_at: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message_id: string;
    message_type: "error" | "response" | "request" | "notification" | "heartbeat";
    sender_id: string;
    payload: Record<string, any>;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    expires_at?: string | undefined;
    receiver_id?: string | undefined;
    reply_to?: string | undefined;
}, {
    timestamp: string;
    message_id: string;
    message_type: "error" | "response" | "request" | "notification" | "heartbeat";
    sender_id: string;
    payload: Record<string, any>;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    expires_at?: string | undefined;
    receiver_id?: string | undefined;
    reply_to?: string | undefined;
}>;
export type AgentMessage = z.infer<typeof AgentMessage>;
/**
 * Agent Definition Schema
 */
export declare const AgentDefinition: z.ZodObject<{
    agent_id: z.ZodString;
    agent_type: z.ZodEnum<["reactive", "proactive", "deliberative", "hybrid"]>;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    version: z.ZodOptional<z.ZodString>;
    capabilities: z.ZodArray<z.ZodString, "many">;
    config: z.ZodOptional<z.ZodObject<{
        max_concurrent_tasks: z.ZodOptional<z.ZodNumber>;
        timeout_ms: z.ZodOptional<z.ZodNumber>;
        retry_policy: z.ZodOptional<z.ZodEnum<["none", "fixed", "exponential"]>>;
        max_retries: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        timeout_ms?: number | undefined;
        max_retries?: number | undefined;
        max_concurrent_tasks?: number | undefined;
        retry_policy?: "none" | "fixed" | "exponential" | undefined;
    }, {
        timeout_ms?: number | undefined;
        max_retries?: number | undefined;
        max_concurrent_tasks?: number | undefined;
        retry_policy?: "none" | "fixed" | "exponential" | undefined;
    }>>;
    communication: z.ZodOptional<z.ZodObject<{
        protocol: z.ZodOptional<z.ZodEnum<["http", "websocket", "amqp", "mqtt"]>>;
        endpoint: z.ZodOptional<z.ZodString>;
        auth_required: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
        endpoint?: string | undefined;
        auth_required?: boolean | undefined;
    }, {
        protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
        endpoint?: string | undefined;
        auth_required?: boolean | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    capabilities: string[];
    agent_type: "hybrid" | "reactive" | "proactive" | "deliberative";
    agent_id: string;
    config?: {
        timeout_ms?: number | undefined;
        max_retries?: number | undefined;
        max_concurrent_tasks?: number | undefined;
        retry_policy?: "none" | "fixed" | "exponential" | undefined;
    } | undefined;
    version?: string | undefined;
    metadata?: Record<string, any> | undefined;
    description?: string | undefined;
    communication?: {
        protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
        endpoint?: string | undefined;
        auth_required?: boolean | undefined;
    } | undefined;
}, {
    name: string;
    capabilities: string[];
    agent_type: "hybrid" | "reactive" | "proactive" | "deliberative";
    agent_id: string;
    config?: {
        timeout_ms?: number | undefined;
        max_retries?: number | undefined;
        max_concurrent_tasks?: number | undefined;
        retry_policy?: "none" | "fixed" | "exponential" | undefined;
    } | undefined;
    version?: string | undefined;
    metadata?: Record<string, any> | undefined;
    description?: string | undefined;
    communication?: {
        protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
        endpoint?: string | undefined;
        auth_required?: boolean | undefined;
    } | undefined;
}>;
export type AgentDefinition = z.infer<typeof AgentDefinition>;
/**
 * Agent JSON Request Schema
 */
export declare const AgentJsonRequest: z.ZodObject<{
    agent_id: z.ZodString;
    action: z.ZodEnum<["send_message", "create_agent", "update_agent", "delete_agent", "query_state", "execute_task"]>;
    message: z.ZodOptional<z.ZodObject<{
        message_id: z.ZodString;
        message_type: z.ZodEnum<["request", "response", "notification", "error", "heartbeat"]>;
        sender_id: z.ZodString;
        receiver_id: z.ZodOptional<z.ZodString>;
        payload: z.ZodRecord<z.ZodString, z.ZodAny>;
        timestamp: z.ZodString;
        correlation_id: z.ZodOptional<z.ZodString>;
        reply_to: z.ZodOptional<z.ZodString>;
        expires_at: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        message_id: string;
        message_type: "error" | "response" | "request" | "notification" | "heartbeat";
        sender_id: string;
        payload: Record<string, any>;
        correlation_id?: string | undefined;
        metadata?: Record<string, any> | undefined;
        expires_at?: string | undefined;
        receiver_id?: string | undefined;
        reply_to?: string | undefined;
    }, {
        timestamp: string;
        message_id: string;
        message_type: "error" | "response" | "request" | "notification" | "heartbeat";
        sender_id: string;
        payload: Record<string, any>;
        correlation_id?: string | undefined;
        metadata?: Record<string, any> | undefined;
        expires_at?: string | undefined;
        receiver_id?: string | undefined;
        reply_to?: string | undefined;
    }>>;
    definition: z.ZodOptional<z.ZodObject<{
        agent_id: z.ZodString;
        agent_type: z.ZodEnum<["reactive", "proactive", "deliberative", "hybrid"]>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        version: z.ZodOptional<z.ZodString>;
        capabilities: z.ZodArray<z.ZodString, "many">;
        config: z.ZodOptional<z.ZodObject<{
            max_concurrent_tasks: z.ZodOptional<z.ZodNumber>;
            timeout_ms: z.ZodOptional<z.ZodNumber>;
            retry_policy: z.ZodOptional<z.ZodEnum<["none", "fixed", "exponential"]>>;
            max_retries: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            timeout_ms?: number | undefined;
            max_retries?: number | undefined;
            max_concurrent_tasks?: number | undefined;
            retry_policy?: "none" | "fixed" | "exponential" | undefined;
        }, {
            timeout_ms?: number | undefined;
            max_retries?: number | undefined;
            max_concurrent_tasks?: number | undefined;
            retry_policy?: "none" | "fixed" | "exponential" | undefined;
        }>>;
        communication: z.ZodOptional<z.ZodObject<{
            protocol: z.ZodOptional<z.ZodEnum<["http", "websocket", "amqp", "mqtt"]>>;
            endpoint: z.ZodOptional<z.ZodString>;
            auth_required: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
            endpoint?: string | undefined;
            auth_required?: boolean | undefined;
        }, {
            protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
            endpoint?: string | undefined;
            auth_required?: boolean | undefined;
        }>>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        capabilities: string[];
        agent_type: "hybrid" | "reactive" | "proactive" | "deliberative";
        agent_id: string;
        config?: {
            timeout_ms?: number | undefined;
            max_retries?: number | undefined;
            max_concurrent_tasks?: number | undefined;
            retry_policy?: "none" | "fixed" | "exponential" | undefined;
        } | undefined;
        version?: string | undefined;
        metadata?: Record<string, any> | undefined;
        description?: string | undefined;
        communication?: {
            protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
            endpoint?: string | undefined;
            auth_required?: boolean | undefined;
        } | undefined;
    }, {
        name: string;
        capabilities: string[];
        agent_type: "hybrid" | "reactive" | "proactive" | "deliberative";
        agent_id: string;
        config?: {
            timeout_ms?: number | undefined;
            max_retries?: number | undefined;
            max_concurrent_tasks?: number | undefined;
            retry_policy?: "none" | "fixed" | "exponential" | undefined;
        } | undefined;
        version?: string | undefined;
        metadata?: Record<string, any> | undefined;
        description?: string | undefined;
        communication?: {
            protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
            endpoint?: string | undefined;
            auth_required?: boolean | undefined;
        } | undefined;
    }>>;
    task: z.ZodOptional<z.ZodObject<{
        task_id: z.ZodOptional<z.ZodString>;
        task_type: z.ZodString;
        parameters: z.ZodRecord<z.ZodString, z.ZodAny>;
        timeout_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        parameters: Record<string, any>;
        task_type: string;
        timeout_ms?: number | undefined;
        task_id?: string | undefined;
    }, {
        parameters: Record<string, any>;
        task_type: string;
        timeout_ms?: number | undefined;
        task_id?: string | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    action: "send_message" | "create_agent" | "update_agent" | "delete_agent" | "query_state" | "execute_task";
    agent_id: string;
    correlation_id?: string | undefined;
    message?: {
        timestamp: string;
        message_id: string;
        message_type: "error" | "response" | "request" | "notification" | "heartbeat";
        sender_id: string;
        payload: Record<string, any>;
        correlation_id?: string | undefined;
        metadata?: Record<string, any> | undefined;
        expires_at?: string | undefined;
        receiver_id?: string | undefined;
        reply_to?: string | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    task?: {
        parameters: Record<string, any>;
        task_type: string;
        timeout_ms?: number | undefined;
        task_id?: string | undefined;
    } | undefined;
    definition?: {
        name: string;
        capabilities: string[];
        agent_type: "hybrid" | "reactive" | "proactive" | "deliberative";
        agent_id: string;
        config?: {
            timeout_ms?: number | undefined;
            max_retries?: number | undefined;
            max_concurrent_tasks?: number | undefined;
            retry_policy?: "none" | "fixed" | "exponential" | undefined;
        } | undefined;
        version?: string | undefined;
        metadata?: Record<string, any> | undefined;
        description?: string | undefined;
        communication?: {
            protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
            endpoint?: string | undefined;
            auth_required?: boolean | undefined;
        } | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    action: "send_message" | "create_agent" | "update_agent" | "delete_agent" | "query_state" | "execute_task";
    agent_id: string;
    correlation_id?: string | undefined;
    message?: {
        timestamp: string;
        message_id: string;
        message_type: "error" | "response" | "request" | "notification" | "heartbeat";
        sender_id: string;
        payload: Record<string, any>;
        correlation_id?: string | undefined;
        metadata?: Record<string, any> | undefined;
        expires_at?: string | undefined;
        receiver_id?: string | undefined;
        reply_to?: string | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    task?: {
        parameters: Record<string, any>;
        task_type: string;
        timeout_ms?: number | undefined;
        task_id?: string | undefined;
    } | undefined;
    definition?: {
        name: string;
        capabilities: string[];
        agent_type: "hybrid" | "reactive" | "proactive" | "deliberative";
        agent_id: string;
        config?: {
            timeout_ms?: number | undefined;
            max_retries?: number | undefined;
            max_concurrent_tasks?: number | undefined;
            retry_policy?: "none" | "fixed" | "exponential" | undefined;
        } | undefined;
        version?: string | undefined;
        metadata?: Record<string, any> | undefined;
        description?: string | undefined;
        communication?: {
            protocol?: "http" | "websocket" | "amqp" | "mqtt" | undefined;
            endpoint?: string | undefined;
            auth_required?: boolean | undefined;
        } | undefined;
    } | undefined;
}>;
export type AgentJsonRequest = z.infer<typeof AgentJsonRequest>;
/**
 * Agent JSON Response Schema
 */
export declare const AgentJsonResponse: z.ZodObject<{
    request_id: z.ZodString;
    action: z.ZodEnum<["send_message", "create_agent", "update_agent", "delete_agent", "query_state", "execute_task"]>;
    status: z.ZodEnum<["success", "failed", "timeout", "partial"]>;
    result: z.ZodOptional<z.ZodObject<{
        message_id: z.ZodOptional<z.ZodString>;
        agent_id: z.ZodOptional<z.ZodString>;
        state: z.ZodOptional<z.ZodEnum<["idle", "processing", "waiting", "error", "terminated"]>>;
        task_result: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        state?: "error" | "idle" | "processing" | "waiting" | "terminated" | undefined;
        agent_id?: string | undefined;
        message_id?: string | undefined;
        task_result?: Record<string, any> | undefined;
    }, {
        state?: "error" | "idle" | "processing" | "waiting" | "terminated" | undefined;
        agent_id?: string | undefined;
        message_id?: string | undefined;
        task_result?: Record<string, any> | undefined;
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
    metadata: z.ZodOptional<z.ZodObject<{
        processing_time_ms: z.ZodOptional<z.ZodNumber>;
        timestamp: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        timestamp?: string | undefined;
        processing_time_ms?: number | undefined;
    }, {
        timestamp?: string | undefined;
        processing_time_ms?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "success" | "failed" | "partial" | "timeout";
    action: "send_message" | "create_agent" | "update_agent" | "delete_agent" | "query_state" | "execute_task";
    request_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        timestamp?: string | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
    result?: {
        state?: "error" | "idle" | "processing" | "waiting" | "terminated" | undefined;
        agent_id?: string | undefined;
        message_id?: string | undefined;
        task_result?: Record<string, any> | undefined;
    } | undefined;
}, {
    timestamp: string;
    status: "success" | "failed" | "partial" | "timeout";
    action: "send_message" | "create_agent" | "update_agent" | "delete_agent" | "query_state" | "execute_task";
    request_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        timestamp?: string | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
    result?: {
        state?: "error" | "idle" | "processing" | "waiting" | "terminated" | undefined;
        agent_id?: string | undefined;
        message_id?: string | undefined;
        task_result?: Record<string, any> | undefined;
    } | undefined;
}>;
export type AgentJsonResponse = z.infer<typeof AgentJsonResponse>;
/**
 * Batch Message Request Schema
 */
export declare const AgentJsonBatchRequest: z.ZodObject<{
    batch_id: z.ZodString;
    messages: z.ZodArray<z.ZodObject<{
        message_id: z.ZodString;
        message_type: z.ZodEnum<["request", "response", "notification", "error", "heartbeat"]>;
        sender_id: z.ZodString;
        receiver_id: z.ZodOptional<z.ZodString>;
        payload: z.ZodRecord<z.ZodString, z.ZodAny>;
        timestamp: z.ZodString;
        correlation_id: z.ZodOptional<z.ZodString>;
        reply_to: z.ZodOptional<z.ZodString>;
        expires_at: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        message_id: string;
        message_type: "error" | "response" | "request" | "notification" | "heartbeat";
        sender_id: string;
        payload: Record<string, any>;
        correlation_id?: string | undefined;
        metadata?: Record<string, any> | undefined;
        expires_at?: string | undefined;
        receiver_id?: string | undefined;
        reply_to?: string | undefined;
    }, {
        timestamp: string;
        message_id: string;
        message_type: "error" | "response" | "request" | "notification" | "heartbeat";
        sender_id: string;
        payload: Record<string, any>;
        correlation_id?: string | undefined;
        metadata?: Record<string, any> | undefined;
        expires_at?: string | undefined;
        receiver_id?: string | undefined;
        reply_to?: string | undefined;
    }>, "many">;
    config: z.ZodOptional<z.ZodObject<{
        parallel: z.ZodOptional<z.ZodBoolean>;
        stop_on_error: z.ZodOptional<z.ZodBoolean>;
        timeout_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        timeout_ms?: number | undefined;
        parallel?: boolean | undefined;
        stop_on_error?: boolean | undefined;
    }, {
        timeout_ms?: number | undefined;
        parallel?: boolean | undefined;
        stop_on_error?: boolean | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    messages: {
        timestamp: string;
        message_id: string;
        message_type: "error" | "response" | "request" | "notification" | "heartbeat";
        sender_id: string;
        payload: Record<string, any>;
        correlation_id?: string | undefined;
        metadata?: Record<string, any> | undefined;
        expires_at?: string | undefined;
        receiver_id?: string | undefined;
        reply_to?: string | undefined;
    }[];
    batch_id: string;
    correlation_id?: string | undefined;
    config?: {
        timeout_ms?: number | undefined;
        parallel?: boolean | undefined;
        stop_on_error?: boolean | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    messages: {
        timestamp: string;
        message_id: string;
        message_type: "error" | "response" | "request" | "notification" | "heartbeat";
        sender_id: string;
        payload: Record<string, any>;
        correlation_id?: string | undefined;
        metadata?: Record<string, any> | undefined;
        expires_at?: string | undefined;
        receiver_id?: string | undefined;
        reply_to?: string | undefined;
    }[];
    batch_id: string;
    correlation_id?: string | undefined;
    config?: {
        timeout_ms?: number | undefined;
        parallel?: boolean | undefined;
        stop_on_error?: boolean | undefined;
    } | undefined;
}>;
export type AgentJsonBatchRequest = z.infer<typeof AgentJsonBatchRequest>;
/**
 * Batch Message Response Schema
 */
export declare const AgentJsonBatchResponse: z.ZodObject<{
    batch_id: z.ZodString;
    status: z.ZodEnum<["completed", "partially_completed", "failed"]>;
    results: z.ZodArray<z.ZodObject<{
        message_id: z.ZodString;
        status: z.ZodEnum<["sent", "failed", "timeout"]>;
        error: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        status: "failed" | "timeout" | "sent";
        message_id: string;
        error?: Record<string, any> | undefined;
    }, {
        status: "failed" | "timeout" | "sent";
        message_id: string;
        error?: Record<string, any> | undefined;
    }>, "many">;
    summary: z.ZodObject<{
        total: z.ZodNumber;
        sent: z.ZodNumber;
        failed: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        failed: number;
        sent: number;
        total: number;
    }, {
        failed: number;
        sent: number;
        total: number;
    }>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "completed" | "failed" | "partially_completed";
    results: {
        status: "failed" | "timeout" | "sent";
        message_id: string;
        error?: Record<string, any> | undefined;
    }[];
    summary: {
        failed: number;
        sent: number;
        total: number;
    };
    batch_id: string;
    correlation_id?: string | undefined;
}, {
    timestamp: string;
    status: "completed" | "failed" | "partially_completed";
    results: {
        status: "failed" | "timeout" | "sent";
        message_id: string;
        error?: Record<string, any> | undefined;
    }[];
    summary: {
        failed: number;
        sent: number;
        total: number;
    };
    batch_id: string;
    correlation_id?: string | undefined;
}>;
export type AgentJsonBatchResponse = z.infer<typeof AgentJsonBatchResponse>;
/**
 * Error Model
 */
export declare const AgentJsonError: z.ZodObject<{
    code: z.ZodEnum<["AGENT_NOT_FOUND", "INVALID_MESSAGE", "AGENT_UNAVAILABLE", "TIMEOUT", "AUTHORIZATION_FAILED", "VALIDATION_ERROR", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "TIMEOUT" | "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "AGENT_NOT_FOUND" | "INVALID_MESSAGE" | "AGENT_UNAVAILABLE" | "AUTHORIZATION_FAILED";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "TIMEOUT" | "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "AGENT_NOT_FOUND" | "INVALID_MESSAGE" | "AGENT_UNAVAILABLE" | "AUTHORIZATION_FAILED";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type AgentJsonError = z.infer<typeof AgentJsonError>;
/**
 * Validation Functions
 */
export declare function validateAgentJsonRequest(data: unknown): {
    success: boolean;
    data?: AgentJsonRequest;
    errors?: string[];
};
export declare function isAgentJsonRequest(data: unknown): data is AgentJsonRequest;
/**
 * Examples
 */
export declare const AgentJsonExamples: {
    validRequest: AgentJsonRequest;
    validResponse: AgentJsonResponse;
};
//# sourceMappingURL=agentjson-canonical.d.ts.map