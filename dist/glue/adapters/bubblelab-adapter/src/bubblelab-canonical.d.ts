/**
 * BubbleLab Canonical Schema
 *
 * Purpose: Define the canonical data model for BubbleLab integration
 * Compliance: Anti-Corruption Layer - normalize BubbleLab data to canonical form
 *
 * This schema maps BubbleLab's bubble/workflow concepts to the OpenEvolve canonical model
 */
import { z } from 'zod';
/**
 * Bubble Type Enumeration
 * Maps to different bubble types in BubbleLab (PostgreSQL, Slack, AI Agent, etc.)
 */
export declare enum BubbleType {
    POSTGRESQL = "postgresql",
    SLACK = "slack",
    AI_AGENT = "ai_agent",
    DATABASE_ANALYZER = "database_analyzer",
    SLACK_NOTIFIER = "slack_notifier",
    WEBHOOK = "webhook",
    CUSTOM = "custom"
}
/**
 * Credential Type Enumeration
 * Maps to BubbleLab credential types
 */
export declare enum CredentialType {
    DATABASE_CRED = "DATABASE_CRED",
    SLACK_CRED = "SLACK_CRED",
    FIRECRAWL_API_KEY = "FIRECRAWL_API_KEY",
    OPENAI_CRED = "OPENAI_CRED",
    ANTHROPIC_CRED = "ANTHROPIC_CRED",
    GOOGLE_GEMINI_CRED = "GOOGLE_GEMINI_CRED"
}
/**
 * Event Type Enumeration
 * Maps to BubbleLab trigger event types
 */
export declare enum EventType {
    WEBHOOK_HTTP = "webhook/http",
    SCHEDULE = "schedule",
    MANUAL = "manual"
}
/**
 * Workflow Execution Status
 */
export declare enum ExecutionStatus {
    PENDING = "pending",
    RUNNING = "running",
    SUCCESS = "success",
    FAILED = "failed",
    TIMEOUT = "timeout"
}
/**
 * Canonical Bubble Definition
 * Represents a single bubble in a workflow
 */
export declare const CanonicalBubbleSchema: z.ZodObject<{
    id: z.ZodOptional<z.ZodString>;
    name: z.ZodString;
    type: z.ZodNativeEnum<typeof BubbleType>;
    config: z.ZodRecord<z.ZodString, z.ZodAny>;
    required_credentials: z.ZodOptional<z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    config: Record<string, any>;
    name: string;
    type: BubbleType;
    metadata?: Record<string, any> | undefined;
    id?: string | undefined;
    required_credentials?: CredentialType[] | undefined;
}, {
    config: Record<string, any>;
    name: string;
    type: BubbleType;
    metadata?: Record<string, any> | undefined;
    id?: string | undefined;
    required_credentials?: CredentialType[] | undefined;
}>;
export type CanonicalBubble = z.infer<typeof CanonicalBubbleSchema>;
/**
 * Canonical BubbleFlow (Workflow) Definition
 * Represents a complete BubbleLab workflow
 */
export declare const CanonicalBubbleFlowSchema: z.ZodObject<{
    id: z.ZodOptional<z.ZodString>;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    event_type: z.ZodNativeEnum<typeof EventType>;
    code: z.ZodOptional<z.ZodString>;
    bubbles: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodOptional<z.ZodString>;
        name: z.ZodString;
        type: z.ZodNativeEnum<typeof BubbleType>;
        config: z.ZodRecord<z.ZodString, z.ZodAny>;
        required_credentials: z.ZodOptional<z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        config: Record<string, any>;
        name: string;
        type: BubbleType;
        metadata?: Record<string, any> | undefined;
        id?: string | undefined;
        required_credentials?: CredentialType[] | undefined;
    }, {
        config: Record<string, any>;
        name: string;
        type: BubbleType;
        metadata?: Record<string, any> | undefined;
        id?: string | undefined;
        required_credentials?: CredentialType[] | undefined;
    }>, "many">>;
    required_credentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">>>;
    webhook_active: z.ZodDefault<z.ZodBoolean>;
    webhook_url: z.ZodOptional<z.ZodString>;
    created_at: z.ZodOptional<z.ZodString>;
    updated_at: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    name: string;
    event_type: EventType;
    webhook_active: boolean;
    id?: string | undefined;
    created_at?: string | undefined;
    code?: string | undefined;
    description?: string | undefined;
    updated_at?: string | undefined;
    required_credentials?: Record<string, CredentialType[]> | undefined;
    bubbles?: {
        config: Record<string, any>;
        name: string;
        type: BubbleType;
        metadata?: Record<string, any> | undefined;
        id?: string | undefined;
        required_credentials?: CredentialType[] | undefined;
    }[] | undefined;
    webhook_url?: string | undefined;
}, {
    name: string;
    event_type: EventType;
    id?: string | undefined;
    created_at?: string | undefined;
    code?: string | undefined;
    description?: string | undefined;
    updated_at?: string | undefined;
    required_credentials?: Record<string, CredentialType[]> | undefined;
    bubbles?: {
        config: Record<string, any>;
        name: string;
        type: BubbleType;
        metadata?: Record<string, any> | undefined;
        id?: string | undefined;
        required_credentials?: CredentialType[] | undefined;
    }[] | undefined;
    webhook_active?: boolean | undefined;
    webhook_url?: string | undefined;
}>;
export type CanonicalBubbleFlow = z.infer<typeof CanonicalBubbleFlowSchema>;
/**
 * Canonical Workflow Execution Result
 */
export declare const CanonicalExecutionResultSchema: z.ZodObject<{
    execution_id: z.ZodOptional<z.ZodString>;
    flow_id: z.ZodString;
    status: z.ZodNativeEnum<typeof ExecutionStatus>;
    output: z.ZodOptional<z.ZodAny>;
    error: z.ZodOptional<z.ZodString>;
    started_at: z.ZodString;
    completed_at: z.ZodOptional<z.ZodString>;
    duration_ms: z.ZodOptional<z.ZodNumber>;
    logs: z.ZodOptional<z.ZodArray<z.ZodObject<{
        timestamp: z.ZodString;
        level: z.ZodString;
        message: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        level: string;
        message: string;
    }, {
        timestamp: string;
        level: string;
        message: string;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    status: ExecutionStatus;
    started_at: string;
    flow_id: string;
    error?: string | undefined;
    duration_ms?: number | undefined;
    execution_id?: string | undefined;
    output?: any;
    completed_at?: string | undefined;
    logs?: {
        timestamp: string;
        level: string;
        message: string;
    }[] | undefined;
}, {
    status: ExecutionStatus;
    started_at: string;
    flow_id: string;
    error?: string | undefined;
    duration_ms?: number | undefined;
    execution_id?: string | undefined;
    output?: any;
    completed_at?: string | undefined;
    logs?: {
        timestamp: string;
        level: string;
        message: string;
    }[] | undefined;
}>;
export type CanonicalExecutionResult = z.infer<typeof CanonicalExecutionResultSchema>;
/**
 * Canonical BubbleLab Event
 * Represents events from BubbleLab to be processed by the orchestration layer
 */
export declare const CanonicalBubbleLabEventSchema: z.ZodObject<{
    event_id: z.ZodString;
    event_type: z.ZodEnum<["workflow.created", "workflow.updated", "workflow.deleted", "workflow.executed", "workflow.execution_failed", "bubble.created", "bubble.updated"]>;
    flow_id: z.ZodOptional<z.ZodString>;
    execution_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
    data: z.ZodAny;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    event_type: "workflow.created" | "workflow.updated" | "workflow.deleted" | "workflow.executed" | "workflow.execution_failed" | "bubble.created" | "bubble.updated";
    event_id: string;
    correlation_id?: string | undefined;
    execution_id?: string | undefined;
    data?: any;
    flow_id?: string | undefined;
}, {
    timestamp: string;
    event_type: "workflow.created" | "workflow.updated" | "workflow.deleted" | "workflow.executed" | "workflow.execution_failed" | "bubble.created" | "bubble.updated";
    event_id: string;
    correlation_id?: string | undefined;
    execution_id?: string | undefined;
    data?: any;
    flow_id?: string | undefined;
}>;
export type CanonicalBubbleLabEvent = z.infer<typeof CanonicalBubbleLabEventSchema>;
/**
 * Canonical Credential Mapping
 * Maps credential types to their IDs
 */
export declare const CanonicalCredentialMappingSchema: z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodNumber>;
export type CanonicalCredentialMapping = z.infer<typeof CanonicalCredentialMappingSchema>;
/**
 * Map BubbleLab API response to Canonical BubbleFlow
 *
 * @param apiResponse - Raw API response from BubbleLab
 * @returns Canonical BubbleFlow
 */
export declare function mapToCanonicalBubbleFlow(apiResponse: any): CanonicalBubbleFlow;
/**
 * Map BubbleLab execution result to canonical form
 */
export declare function mapToCanonicalExecutionResult(apiResponse: any, flowId: string): CanonicalExecutionResult;
/**
 * Map Canonical BubbleFlow to BubbleLab API request format
 */
export declare function mapFromCanonicalBubbleFlow(canonical: CanonicalBubbleFlow): any;
/**
 * Map Canonical Credential Mapping to BubbleLab format
 */
export declare function mapFromCanonicalCredentials(canonical: CanonicalCredentialMapping): Record<string, number>;
/**
 * Validate and parse a CanonicalBubbleFlow
 */
export declare function validateCanonicalBubbleFlow(data: unknown): CanonicalBubbleFlow;
/**
 * Validate and parse a CanonicalExecutionResult
 */
export declare function validateCanonicalExecutionResult(data: unknown): CanonicalExecutionResult;
/**
 * Validate and parse a CanonicalBubbleLabEvent
 */
export declare function validateCanonicalBubbleLabEvent(data: unknown): CanonicalBubbleLabEvent;
/**
 * Generate a unique correlation ID for tracking
 */
export declare function generateCorrelationId(): string;
/**
 * Convert UTC Date to ISO-8601 string (Law of UTC)
 */
export declare function toUTCISOString(date: Date): string;
/**
 * Parse ISO-8601 string to UTC Date
 */
export declare function fromUTCISOString(isoString: string): Date;
//# sourceMappingURL=bubblelab-canonical.d.ts.map