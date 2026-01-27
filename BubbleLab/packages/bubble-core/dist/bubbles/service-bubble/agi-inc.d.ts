import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const AGIIncParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"create_session">;
    agent_name: z.ZodDefault<z.ZodOptional<z.ZodEnum<["agi-0", "agi-0-fast"]>>>;
    webhook_url: z.ZodOptional<z.ZodString>;
    restore_from_session_id: z.ZodOptional<z.ZodString>;
    restore_default_environment_from_user_id: z.ZodOptional<z.ZodString>;
    enable_memory_snapshot: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_session";
    agent_name: "agi-0" | "agi-0-fast";
    enable_memory_snapshot: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    webhook_url?: string | undefined;
    restore_from_session_id?: string | undefined;
    restore_default_environment_from_user_id?: string | undefined;
}, {
    operation: "create_session";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    agent_name?: "agi-0" | "agi-0-fast" | undefined;
    webhook_url?: string | undefined;
    restore_from_session_id?: string | undefined;
    restore_default_environment_from_user_id?: string | undefined;
    enable_memory_snapshot?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_sessions">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_sessions";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "list_sessions";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_session">;
    session_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_session">;
    session_id: z.ZodString;
    save_snapshot_mode: z.ZodDefault<z.ZodOptional<z.ZodEnum<["none", "memory", "filesystem"]>>>;
    save_as_default: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_session";
    session_id: string;
    save_snapshot_mode: "none" | "memory" | "filesystem";
    save_as_default: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    save_snapshot_mode?: "none" | "memory" | "filesystem" | undefined;
    save_as_default?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_all_sessions">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_all_sessions";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_all_sessions";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_message">;
    session_id: z.ZodString;
    message: z.ZodString;
    start_url: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    message: string;
    operation: "send_message";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    start_url?: string | undefined;
}, {
    message: string;
    operation: "send_message";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    start_url?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_status">;
    session_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_status";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_status";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_messages">;
    session_id: z.ZodString;
    after_id: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    sanitize: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_messages";
    session_id: string;
    after_id: number;
    sanitize: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_messages";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    after_id?: number | undefined;
    sanitize?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"pause_session">;
    session_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "pause_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "pause_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"resume_session">;
    session_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "resume_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "resume_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"cancel_session">;
    session_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "cancel_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "cancel_session";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"navigate">;
    session_id: z.ZodString;
    url: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    url: string;
    operation: "navigate";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    url: string;
    operation: "navigate";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_screenshot">;
    session_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_screenshot";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_screenshot";
    session_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
declare const AGIIncResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"create_session">;
    ok: z.ZodBoolean;
    session_id: z.ZodOptional<z.ZodString>;
    vnc_url: z.ZodOptional<z.ZodString>;
    agent_name: z.ZodOptional<z.ZodString>;
    status: z.ZodOptional<z.ZodEnum<["initializing", "ready", "running", "paused", "completed", "error", "terminated"]>>;
    created_at: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_session";
    ok: boolean;
    status?: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated" | undefined;
    created_at?: string | undefined;
    agent_name?: string | undefined;
    session_id?: string | undefined;
    vnc_url?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_session";
    ok: boolean;
    status?: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated" | undefined;
    created_at?: string | undefined;
    agent_name?: string | undefined;
    session_id?: string | undefined;
    vnc_url?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_sessions">;
    ok: z.ZodBoolean;
    sessions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        session_id: z.ZodString;
        vnc_url: z.ZodOptional<z.ZodString>;
        agent_name: z.ZodString;
        status: z.ZodEnum<["initializing", "ready", "running", "paused", "completed", "error", "terminated"]>;
        created_at: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
        created_at: string;
        agent_name: string;
        session_id: string;
        vnc_url?: string | undefined;
    }, {
        status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
        created_at: string;
        agent_name: string;
        session_id: string;
        vnc_url?: string | undefined;
    }>, "many">>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_sessions";
    ok: boolean;
    sessions?: {
        status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
        created_at: string;
        agent_name: string;
        session_id: string;
        vnc_url?: string | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_sessions";
    ok: boolean;
    sessions?: {
        status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
        created_at: string;
        agent_name: string;
        session_id: string;
        vnc_url?: string | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_session">;
    ok: z.ZodBoolean;
    session: z.ZodOptional<z.ZodObject<{
        session_id: z.ZodString;
        vnc_url: z.ZodOptional<z.ZodString>;
        agent_name: z.ZodString;
        status: z.ZodEnum<["initializing", "ready", "running", "paused", "completed", "error", "terminated"]>;
        created_at: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
        created_at: string;
        agent_name: string;
        session_id: string;
        vnc_url?: string | undefined;
    }, {
        status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
        created_at: string;
        agent_name: string;
        session_id: string;
        vnc_url?: string | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_session";
    ok: boolean;
    session?: {
        status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
        created_at: string;
        agent_name: string;
        session_id: string;
        vnc_url?: string | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_session";
    ok: boolean;
    session?: {
        status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
        created_at: string;
        agent_name: string;
        session_id: string;
        vnc_url?: string | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_session">;
    ok: z.ZodBoolean;
    deleted: z.ZodOptional<z.ZodBoolean>;
    message: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_session";
    ok: boolean;
    message?: string | undefined;
    deleted?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_session";
    ok: boolean;
    message?: string | undefined;
    deleted?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_all_sessions">;
    ok: z.ZodBoolean;
    deleted: z.ZodOptional<z.ZodBoolean>;
    message: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_all_sessions";
    ok: boolean;
    message?: string | undefined;
    deleted?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_all_sessions";
    ok: boolean;
    message?: string | undefined;
    deleted?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_message">;
    ok: z.ZodBoolean;
    message: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "send_message";
    ok: boolean;
    message?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "send_message";
    ok: boolean;
    message?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_status">;
    ok: z.ZodBoolean;
    status: z.ZodOptional<z.ZodEnum<["running", "waiting_for_input", "finished", "error"]>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_status";
    ok: boolean;
    status?: "error" | "running" | "waiting_for_input" | "finished" | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_status";
    ok: boolean;
    status?: "error" | "running" | "waiting_for_input" | "finished" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_messages">;
    ok: z.ZodBoolean;
    messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        type: z.ZodEnum<["THOUGHT", "QUESTION", "USER", "DONE", "ERROR", "LOG"]>;
        content: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
        timestamp: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        type: "THOUGHT" | "QUESTION" | "USER" | "DONE" | "ERROR" | "LOG";
        content: string | Record<string, unknown>;
        timestamp: string;
        id: number;
        metadata?: Record<string, unknown> | undefined;
    }, {
        type: "THOUGHT" | "QUESTION" | "USER" | "DONE" | "ERROR" | "LOG";
        content: string | Record<string, unknown>;
        timestamp: string;
        id: number;
        metadata?: Record<string, unknown> | undefined;
    }>, "many">>;
    status: z.ZodOptional<z.ZodEnum<["running", "waiting_for_input", "finished", "error"]>>;
    has_agent: z.ZodOptional<z.ZodBoolean>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_messages";
    ok: boolean;
    status?: "error" | "running" | "waiting_for_input" | "finished" | undefined;
    messages?: {
        type: "THOUGHT" | "QUESTION" | "USER" | "DONE" | "ERROR" | "LOG";
        content: string | Record<string, unknown>;
        timestamp: string;
        id: number;
        metadata?: Record<string, unknown> | undefined;
    }[] | undefined;
    has_agent?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_messages";
    ok: boolean;
    status?: "error" | "running" | "waiting_for_input" | "finished" | undefined;
    messages?: {
        type: "THOUGHT" | "QUESTION" | "USER" | "DONE" | "ERROR" | "LOG";
        content: string | Record<string, unknown>;
        timestamp: string;
        id: number;
        metadata?: Record<string, unknown> | undefined;
    }[] | undefined;
    has_agent?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"pause_session">;
    ok: z.ZodBoolean;
    message: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "pause_session";
    ok: boolean;
    message?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "pause_session";
    ok: boolean;
    message?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"resume_session">;
    ok: z.ZodBoolean;
    message: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "resume_session";
    ok: boolean;
    message?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "resume_session";
    ok: boolean;
    message?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"cancel_session">;
    ok: z.ZodBoolean;
    message: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "cancel_session";
    ok: boolean;
    message?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "cancel_session";
    ok: boolean;
    message?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"navigate">;
    ok: z.ZodBoolean;
    current_url: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "navigate";
    ok: boolean;
    current_url?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "navigate";
    ok: boolean;
    current_url?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_screenshot">;
    ok: z.ZodBoolean;
    screenshot: z.ZodOptional<z.ZodString>;
    url: z.ZodOptional<z.ZodString>;
    title: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_screenshot";
    ok: boolean;
    title?: string | undefined;
    url?: string | undefined;
    screenshot?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_screenshot";
    ok: boolean;
    title?: string | undefined;
    url?: string | undefined;
    screenshot?: string | undefined;
}>]>;
type AGIIncResult = z.output<typeof AGIIncResultSchema>;
type AGIIncParams = z.input<typeof AGIIncParamsSchema>;
export type AGIIncParamsInput = z.input<typeof AGIIncParamsSchema>;
export type AGIIncOperationResult<T extends AGIIncParams['operation']> = Extract<AGIIncResult, {
    operation: T;
}>;
export declare class AGIIncBubble<T extends AGIIncParams = AGIIncParams> extends ServiceBubble<T, Extract<AGIIncResult, {
    operation: T['operation'];
}>> {
    testCredential(): Promise<boolean>;
    static readonly type: "service";
    static readonly service = "agi-inc";
    static readonly authType: "apikey";
    static readonly bubbleName = "agi-inc";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"create_session">;
        agent_name: z.ZodDefault<z.ZodOptional<z.ZodEnum<["agi-0", "agi-0-fast"]>>>;
        webhook_url: z.ZodOptional<z.ZodString>;
        restore_from_session_id: z.ZodOptional<z.ZodString>;
        restore_default_environment_from_user_id: z.ZodOptional<z.ZodString>;
        enable_memory_snapshot: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_session";
        agent_name: "agi-0" | "agi-0-fast";
        enable_memory_snapshot: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        webhook_url?: string | undefined;
        restore_from_session_id?: string | undefined;
        restore_default_environment_from_user_id?: string | undefined;
    }, {
        operation: "create_session";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        agent_name?: "agi-0" | "agi-0-fast" | undefined;
        webhook_url?: string | undefined;
        restore_from_session_id?: string | undefined;
        restore_default_environment_from_user_id?: string | undefined;
        enable_memory_snapshot?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_sessions">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_sessions";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "list_sessions";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_session">;
        session_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_session">;
        session_id: z.ZodString;
        save_snapshot_mode: z.ZodDefault<z.ZodOptional<z.ZodEnum<["none", "memory", "filesystem"]>>>;
        save_as_default: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_session";
        session_id: string;
        save_snapshot_mode: "none" | "memory" | "filesystem";
        save_as_default: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        save_snapshot_mode?: "none" | "memory" | "filesystem" | undefined;
        save_as_default?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_all_sessions">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_all_sessions";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_all_sessions";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_message">;
        session_id: z.ZodString;
        message: z.ZodString;
        start_url: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        operation: "send_message";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        start_url?: string | undefined;
    }, {
        message: string;
        operation: "send_message";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        start_url?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_status">;
        session_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_status";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_status";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_messages">;
        session_id: z.ZodString;
        after_id: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        sanitize: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_messages";
        session_id: string;
        after_id: number;
        sanitize: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_messages";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        after_id?: number | undefined;
        sanitize?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"pause_session">;
        session_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "pause_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "pause_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"resume_session">;
        session_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "resume_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "resume_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"cancel_session">;
        session_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "cancel_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "cancel_session";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"navigate">;
        session_id: z.ZodString;
        url: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        url: string;
        operation: "navigate";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        url: string;
        operation: "navigate";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_screenshot">;
        session_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_screenshot";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_screenshot";
        session_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"create_session">;
        ok: z.ZodBoolean;
        session_id: z.ZodOptional<z.ZodString>;
        vnc_url: z.ZodOptional<z.ZodString>;
        agent_name: z.ZodOptional<z.ZodString>;
        status: z.ZodOptional<z.ZodEnum<["initializing", "ready", "running", "paused", "completed", "error", "terminated"]>>;
        created_at: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_session";
        ok: boolean;
        status?: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated" | undefined;
        created_at?: string | undefined;
        agent_name?: string | undefined;
        session_id?: string | undefined;
        vnc_url?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_session";
        ok: boolean;
        status?: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated" | undefined;
        created_at?: string | undefined;
        agent_name?: string | undefined;
        session_id?: string | undefined;
        vnc_url?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_sessions">;
        ok: z.ZodBoolean;
        sessions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            session_id: z.ZodString;
            vnc_url: z.ZodOptional<z.ZodString>;
            agent_name: z.ZodString;
            status: z.ZodEnum<["initializing", "ready", "running", "paused", "completed", "error", "terminated"]>;
            created_at: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
            created_at: string;
            agent_name: string;
            session_id: string;
            vnc_url?: string | undefined;
        }, {
            status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
            created_at: string;
            agent_name: string;
            session_id: string;
            vnc_url?: string | undefined;
        }>, "many">>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_sessions";
        ok: boolean;
        sessions?: {
            status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
            created_at: string;
            agent_name: string;
            session_id: string;
            vnc_url?: string | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_sessions";
        ok: boolean;
        sessions?: {
            status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
            created_at: string;
            agent_name: string;
            session_id: string;
            vnc_url?: string | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_session">;
        ok: z.ZodBoolean;
        session: z.ZodOptional<z.ZodObject<{
            session_id: z.ZodString;
            vnc_url: z.ZodOptional<z.ZodString>;
            agent_name: z.ZodString;
            status: z.ZodEnum<["initializing", "ready", "running", "paused", "completed", "error", "terminated"]>;
            created_at: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
            created_at: string;
            agent_name: string;
            session_id: string;
            vnc_url?: string | undefined;
        }, {
            status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
            created_at: string;
            agent_name: string;
            session_id: string;
            vnc_url?: string | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_session";
        ok: boolean;
        session?: {
            status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
            created_at: string;
            agent_name: string;
            session_id: string;
            vnc_url?: string | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_session";
        ok: boolean;
        session?: {
            status: "error" | "completed" | "initializing" | "ready" | "running" | "paused" | "terminated";
            created_at: string;
            agent_name: string;
            session_id: string;
            vnc_url?: string | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_session">;
        ok: z.ZodBoolean;
        deleted: z.ZodOptional<z.ZodBoolean>;
        message: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_session";
        ok: boolean;
        message?: string | undefined;
        deleted?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_session";
        ok: boolean;
        message?: string | undefined;
        deleted?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_all_sessions">;
        ok: z.ZodBoolean;
        deleted: z.ZodOptional<z.ZodBoolean>;
        message: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_all_sessions";
        ok: boolean;
        message?: string | undefined;
        deleted?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_all_sessions";
        ok: boolean;
        message?: string | undefined;
        deleted?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_message">;
        ok: z.ZodBoolean;
        message: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "send_message";
        ok: boolean;
        message?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "send_message";
        ok: boolean;
        message?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_status">;
        ok: z.ZodBoolean;
        status: z.ZodOptional<z.ZodEnum<["running", "waiting_for_input", "finished", "error"]>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_status";
        ok: boolean;
        status?: "error" | "running" | "waiting_for_input" | "finished" | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_status";
        ok: boolean;
        status?: "error" | "running" | "waiting_for_input" | "finished" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_messages">;
        ok: z.ZodBoolean;
        messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            type: z.ZodEnum<["THOUGHT", "QUESTION", "USER", "DONE", "ERROR", "LOG"]>;
            content: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
            timestamp: z.ZodString;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            type: "THOUGHT" | "QUESTION" | "USER" | "DONE" | "ERROR" | "LOG";
            content: string | Record<string, unknown>;
            timestamp: string;
            id: number;
            metadata?: Record<string, unknown> | undefined;
        }, {
            type: "THOUGHT" | "QUESTION" | "USER" | "DONE" | "ERROR" | "LOG";
            content: string | Record<string, unknown>;
            timestamp: string;
            id: number;
            metadata?: Record<string, unknown> | undefined;
        }>, "many">>;
        status: z.ZodOptional<z.ZodEnum<["running", "waiting_for_input", "finished", "error"]>>;
        has_agent: z.ZodOptional<z.ZodBoolean>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_messages";
        ok: boolean;
        status?: "error" | "running" | "waiting_for_input" | "finished" | undefined;
        messages?: {
            type: "THOUGHT" | "QUESTION" | "USER" | "DONE" | "ERROR" | "LOG";
            content: string | Record<string, unknown>;
            timestamp: string;
            id: number;
            metadata?: Record<string, unknown> | undefined;
        }[] | undefined;
        has_agent?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_messages";
        ok: boolean;
        status?: "error" | "running" | "waiting_for_input" | "finished" | undefined;
        messages?: {
            type: "THOUGHT" | "QUESTION" | "USER" | "DONE" | "ERROR" | "LOG";
            content: string | Record<string, unknown>;
            timestamp: string;
            id: number;
            metadata?: Record<string, unknown> | undefined;
        }[] | undefined;
        has_agent?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"pause_session">;
        ok: z.ZodBoolean;
        message: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "pause_session";
        ok: boolean;
        message?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "pause_session";
        ok: boolean;
        message?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"resume_session">;
        ok: z.ZodBoolean;
        message: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "resume_session";
        ok: boolean;
        message?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "resume_session";
        ok: boolean;
        message?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"cancel_session">;
        ok: z.ZodBoolean;
        message: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "cancel_session";
        ok: boolean;
        message?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "cancel_session";
        ok: boolean;
        message?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"navigate">;
        ok: z.ZodBoolean;
        current_url: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "navigate";
        ok: boolean;
        current_url?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "navigate";
        ok: boolean;
        current_url?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_screenshot">;
        ok: z.ZodBoolean;
        screenshot: z.ZodOptional<z.ZodString>;
        url: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_screenshot";
        ok: boolean;
        title?: string | undefined;
        url?: string | undefined;
        screenshot?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_screenshot";
        ok: boolean;
        title?: string | undefined;
        url?: string | undefined;
        screenshot?: string | undefined;
    }>]>;
    static readonly shortDescription = "AGI Agent integration for browser automation and task execution";
    static readonly longDescription = "\n    AGI Agent Sessions API integration for creating browser agents that can perform tasks autonomously.\n    Use cases:\n    - Internet research and data extraction\n    - Form filling and web automation\n    - Making purchases with guest checkout\n    - General web automation tasks\n\n    Features:\n    - Create and manage browser sessions\n    - Send tasks and monitor progress\n    - Control execution (pause, resume, cancel)\n    - Capture screenshots\n    - Webhook support for real-time updates\n\n    Security Features:\n    - Bearer token authentication\n    - Rate limiting protection\n    - Session isolation\n    - Comprehensive error handling\n  ";
    static readonly alias = "agi-inc";
    constructor(params?: T, context?: BubbleContext, instanceId?: string);
    protected performAction(context?: BubbleContext): Promise<Extract<AGIIncResult, {
        operation: T['operation'];
    }>>;
    private createSession;
    private listSessions;
    private getSession;
    private deleteSession;
    private deleteAllSessions;
    private sendMessage;
    private getStatus;
    private getMessages;
    private pauseSession;
    private resumeSession;
    private cancelSession;
    private navigate;
    private getScreenshot;
    protected chooseCredential(): string | undefined;
    private makeAGIApiCall;
}
export {};
//# sourceMappingURL=agi-inc.d.ts.map