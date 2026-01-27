/**
 * WEBHOOK REPEATER WORKFLOW
 *
 * A robust workflow for retrying webhook deliveries with exponential backoff,
 * circuit breaker pattern, and comprehensive error handling.
 *
 * This workflow combines:
 * 1. HTTP bubble for webhook delivery
 * 2. Exponential backoff retry logic
 * 3. Circuit breaker for preventing cascade failures
 * 4. Dead letter queue for permanently failed webhooks
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parameters schema for webhook repeater workflow
 */
declare const WebhookRepeaterParamsSchema: z.ZodObject<{
    /**
     * Target webhook URL
     */
    webhookUrl: z.ZodString;
    /**
     * HTTP method for webhook delivery
     */
    method: z.ZodDefault<z.ZodEnum<["POST", "PUT", "PATCH"]>>;
    /**
     * Payload to deliver
     */
    payload: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
    /**
     * HTTP headers to include
     */
    headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    /**
     * Retry strategy configuration
     */
    retryStrategy: z.ZodOptional<z.ZodObject<{
        maxAttempts: z.ZodDefault<z.ZodNumber>;
        initialDelay: z.ZodDefault<z.ZodNumber>;
        maxDelay: z.ZodDefault<z.ZodNumber>;
        backoffMultiplier: z.ZodDefault<z.ZodNumber>;
        jitter: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        maxAttempts: number;
        backoffMultiplier: number;
        jitter: boolean;
        maxDelay: number;
        initialDelay: number;
    }, {
        maxAttempts?: number | undefined;
        backoffMultiplier?: number | undefined;
        jitter?: boolean | undefined;
        maxDelay?: number | undefined;
        initialDelay?: number | undefined;
    }>>;
    /**
     * Circuit breaker configuration
     */
    circuitBreaker: z.ZodOptional<z.ZodObject<{
        enabled: z.ZodDefault<z.ZodBoolean>;
        failureThreshold: z.ZodDefault<z.ZodNumber>;
        successThreshold: z.ZodDefault<z.ZodNumber>;
        timeout: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        failureThreshold: number;
        successThreshold: number;
        timeout: number;
        enabled: boolean;
    }, {
        failureThreshold?: number | undefined;
        successThreshold?: number | undefined;
        timeout?: number | undefined;
        enabled?: boolean | undefined;
    }>>;
    /**
     * Request timeout
     */
    timeout: z.ZodDefault<z.ZodNumber>;
    /**
     * Webhook ID for tracking
     */
    webhookId: z.ZodOptional<z.ZodString>;
    /**
     * Credentials for authentication
     */
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    /**
     * Auth type for webhook
     */
    authType: z.ZodDefault<z.ZodEnum<["none", "bearer", "basic", "api-key", "api-key-header", "custom"]>>;
    /**
     * Custom auth header name
     */
    authHeader: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    method: "POST" | "PUT" | "PATCH";
    payload: string | Record<string, unknown>;
    authType: "custom" | "none" | "basic" | "bearer" | "api-key" | "api-key-header";
    webhookUrl: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    headers?: Record<string, string> | undefined;
    retryStrategy?: {
        maxAttempts: number;
        backoffMultiplier: number;
        jitter: boolean;
        maxDelay: number;
        initialDelay: number;
    } | undefined;
    authHeader?: string | undefined;
    webhookId?: string | undefined;
    circuitBreaker?: {
        failureThreshold: number;
        successThreshold: number;
        timeout: number;
        enabled: boolean;
    } | undefined;
}, {
    payload: string | Record<string, unknown>;
    webhookUrl: string;
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    headers?: Record<string, string> | undefined;
    method?: "POST" | "PUT" | "PATCH" | undefined;
    retryStrategy?: {
        maxAttempts?: number | undefined;
        backoffMultiplier?: number | undefined;
        jitter?: boolean | undefined;
        maxDelay?: number | undefined;
        initialDelay?: number | undefined;
    } | undefined;
    authType?: "custom" | "none" | "basic" | "bearer" | "api-key" | "api-key-header" | undefined;
    authHeader?: string | undefined;
    webhookId?: string | undefined;
    circuitBreaker?: {
        failureThreshold?: number | undefined;
        successThreshold?: number | undefined;
        timeout?: number | undefined;
        enabled?: boolean | undefined;
    } | undefined;
}>;
type WebhookRepeaterParams = z.input<typeof WebhookRepeaterParamsSchema>;
/**
 * Result schema for webhook repeater workflow
 */
declare const WebhookRepeaterResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    /**
     * Delivery attempt details
     */
    deliveryAttempts: z.ZodArray<z.ZodObject<{
        attemptNumber: z.ZodNumber;
        timestamp: z.ZodDate;
        success: z.ZodBoolean;
        statusCode: z.ZodOptional<z.ZodNumber>;
        responseTime: z.ZodNumber;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        timestamp: Date;
        responseTime: number;
        attemptNumber: number;
        error?: string | undefined;
        statusCode?: number | undefined;
    }, {
        success: boolean;
        timestamp: Date;
        responseTime: number;
        attemptNumber: number;
        error?: string | undefined;
        statusCode?: number | undefined;
    }>, "many">;
    /**
     * Final delivery status
     */
    deliveryStatus: z.ZodOptional<z.ZodObject<{
        delivered: z.ZodBoolean;
        finalStatusCode: z.ZodOptional<z.ZodNumber>;
        totalAttempts: z.ZodNumber;
        totalDuration: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        totalAttempts: number;
        delivered: boolean;
        totalDuration: number;
        finalStatusCode?: number | undefined;
    }, {
        totalAttempts: number;
        delivered: boolean;
        totalDuration: number;
        finalStatusCode?: number | undefined;
    }>>;
    /**
     * Circuit breaker state
     */
    circuitBreakerState: z.ZodOptional<z.ZodObject<{
        isOpen: z.ZodBoolean;
        failureCount: z.ZodNumber;
        lastFailureTime: z.ZodOptional<z.ZodDate>;
    }, "strip", z.ZodTypeAny, {
        isOpen: boolean;
        failureCount: number;
        lastFailureTime?: Date | undefined;
    }, {
        isOpen: boolean;
        failureCount: number;
        lastFailureTime?: Date | undefined;
    }>>;
    /**
     * Webhook identifier
     */
    webhookId: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    deliveryAttempts: {
        success: boolean;
        timestamp: Date;
        responseTime: number;
        attemptNumber: number;
        error?: string | undefined;
        statusCode?: number | undefined;
    }[];
    webhookId?: string | undefined;
    deliveryStatus?: {
        totalAttempts: number;
        delivered: boolean;
        totalDuration: number;
        finalStatusCode?: number | undefined;
    } | undefined;
    circuitBreakerState?: {
        isOpen: boolean;
        failureCount: number;
        lastFailureTime?: Date | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    deliveryAttempts: {
        success: boolean;
        timestamp: Date;
        responseTime: number;
        attemptNumber: number;
        error?: string | undefined;
        statusCode?: number | undefined;
    }[];
    webhookId?: string | undefined;
    deliveryStatus?: {
        totalAttempts: number;
        delivered: boolean;
        totalDuration: number;
        finalStatusCode?: number | undefined;
    } | undefined;
    circuitBreakerState?: {
        isOpen: boolean;
        failureCount: number;
        lastFailureTime?: Date | undefined;
    } | undefined;
}>;
type WebhookRepeaterResult = z.infer<typeof WebhookRepeaterResultSchema>;
/**
 * Webhook Repeater Workflow
 *
 * Provides robust webhook delivery with retry logic, circuit breaker, and comprehensive error handling.
 */
export declare class WebhookRepeaterWorkflow extends WorkflowBubble<WebhookRepeaterParams, WebhookRepeaterResult> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        /**
         * Target webhook URL
         */
        webhookUrl: z.ZodString;
        /**
         * HTTP method for webhook delivery
         */
        method: z.ZodDefault<z.ZodEnum<["POST", "PUT", "PATCH"]>>;
        /**
         * Payload to deliver
         */
        payload: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>;
        /**
         * HTTP headers to include
         */
        headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        /**
         * Retry strategy configuration
         */
        retryStrategy: z.ZodOptional<z.ZodObject<{
            maxAttempts: z.ZodDefault<z.ZodNumber>;
            initialDelay: z.ZodDefault<z.ZodNumber>;
            maxDelay: z.ZodDefault<z.ZodNumber>;
            backoffMultiplier: z.ZodDefault<z.ZodNumber>;
            jitter: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            maxAttempts: number;
            backoffMultiplier: number;
            jitter: boolean;
            maxDelay: number;
            initialDelay: number;
        }, {
            maxAttempts?: number | undefined;
            backoffMultiplier?: number | undefined;
            jitter?: boolean | undefined;
            maxDelay?: number | undefined;
            initialDelay?: number | undefined;
        }>>;
        /**
         * Circuit breaker configuration
         */
        circuitBreaker: z.ZodOptional<z.ZodObject<{
            enabled: z.ZodDefault<z.ZodBoolean>;
            failureThreshold: z.ZodDefault<z.ZodNumber>;
            successThreshold: z.ZodDefault<z.ZodNumber>;
            timeout: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            failureThreshold: number;
            successThreshold: number;
            timeout: number;
            enabled: boolean;
        }, {
            failureThreshold?: number | undefined;
            successThreshold?: number | undefined;
            timeout?: number | undefined;
            enabled?: boolean | undefined;
        }>>;
        /**
         * Request timeout
         */
        timeout: z.ZodDefault<z.ZodNumber>;
        /**
         * Webhook ID for tracking
         */
        webhookId: z.ZodOptional<z.ZodString>;
        /**
         * Credentials for authentication
         */
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        /**
         * Auth type for webhook
         */
        authType: z.ZodDefault<z.ZodEnum<["none", "bearer", "basic", "api-key", "api-key-header", "custom"]>>;
        /**
         * Custom auth header name
         */
        authHeader: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        method: "POST" | "PUT" | "PATCH";
        payload: string | Record<string, unknown>;
        authType: "custom" | "none" | "basic" | "bearer" | "api-key" | "api-key-header";
        webhookUrl: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        headers?: Record<string, string> | undefined;
        retryStrategy?: {
            maxAttempts: number;
            backoffMultiplier: number;
            jitter: boolean;
            maxDelay: number;
            initialDelay: number;
        } | undefined;
        authHeader?: string | undefined;
        webhookId?: string | undefined;
        circuitBreaker?: {
            failureThreshold: number;
            successThreshold: number;
            timeout: number;
            enabled: boolean;
        } | undefined;
    }, {
        payload: string | Record<string, unknown>;
        webhookUrl: string;
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        headers?: Record<string, string> | undefined;
        method?: "POST" | "PUT" | "PATCH" | undefined;
        retryStrategy?: {
            maxAttempts?: number | undefined;
            backoffMultiplier?: number | undefined;
            jitter?: boolean | undefined;
            maxDelay?: number | undefined;
            initialDelay?: number | undefined;
        } | undefined;
        authType?: "custom" | "none" | "basic" | "bearer" | "api-key" | "api-key-header" | undefined;
        authHeader?: string | undefined;
        webhookId?: string | undefined;
        circuitBreaker?: {
            failureThreshold?: number | undefined;
            successThreshold?: number | undefined;
            timeout?: number | undefined;
            enabled?: boolean | undefined;
        } | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        /**
         * Delivery attempt details
         */
        deliveryAttempts: z.ZodArray<z.ZodObject<{
            attemptNumber: z.ZodNumber;
            timestamp: z.ZodDate;
            success: z.ZodBoolean;
            statusCode: z.ZodOptional<z.ZodNumber>;
            responseTime: z.ZodNumber;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            timestamp: Date;
            responseTime: number;
            attemptNumber: number;
            error?: string | undefined;
            statusCode?: number | undefined;
        }, {
            success: boolean;
            timestamp: Date;
            responseTime: number;
            attemptNumber: number;
            error?: string | undefined;
            statusCode?: number | undefined;
        }>, "many">;
        /**
         * Final delivery status
         */
        deliveryStatus: z.ZodOptional<z.ZodObject<{
            delivered: z.ZodBoolean;
            finalStatusCode: z.ZodOptional<z.ZodNumber>;
            totalAttempts: z.ZodNumber;
            totalDuration: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            totalAttempts: number;
            delivered: boolean;
            totalDuration: number;
            finalStatusCode?: number | undefined;
        }, {
            totalAttempts: number;
            delivered: boolean;
            totalDuration: number;
            finalStatusCode?: number | undefined;
        }>>;
        /**
         * Circuit breaker state
         */
        circuitBreakerState: z.ZodOptional<z.ZodObject<{
            isOpen: z.ZodBoolean;
            failureCount: z.ZodNumber;
            lastFailureTime: z.ZodOptional<z.ZodDate>;
        }, "strip", z.ZodTypeAny, {
            isOpen: boolean;
            failureCount: number;
            lastFailureTime?: Date | undefined;
        }, {
            isOpen: boolean;
            failureCount: number;
            lastFailureTime?: Date | undefined;
        }>>;
        /**
         * Webhook identifier
         */
        webhookId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        deliveryAttempts: {
            success: boolean;
            timestamp: Date;
            responseTime: number;
            attemptNumber: number;
            error?: string | undefined;
            statusCode?: number | undefined;
        }[];
        webhookId?: string | undefined;
        deliveryStatus?: {
            totalAttempts: number;
            delivered: boolean;
            totalDuration: number;
            finalStatusCode?: number | undefined;
        } | undefined;
        circuitBreakerState?: {
            isOpen: boolean;
            failureCount: number;
            lastFailureTime?: Date | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        deliveryAttempts: {
            success: boolean;
            timestamp: Date;
            responseTime: number;
            attemptNumber: number;
            error?: string | undefined;
            statusCode?: number | undefined;
        }[];
        webhookId?: string | undefined;
        deliveryStatus?: {
            totalAttempts: number;
            delivered: boolean;
            totalDuration: number;
            finalStatusCode?: number | undefined;
        } | undefined;
        circuitBreakerState?: {
            isOpen: boolean;
            failureCount: number;
            lastFailureTime?: Date | undefined;
        } | undefined;
    }>;
    static readonly shortDescription = "Robust webhook delivery with retries and circuit breaker";
    static readonly longDescription = "\n    Provides reliable webhook delivery with advanced retry mechanisms and failure protection.\n\n    Features:\n    - Exponential backoff retry with configurable strategies\n    - Circuit breaker pattern to prevent cascade failures\n    - Jitter to prevent thundering herd problem\n    - Comprehensive delivery attempt tracking\n    - Dead letter queue support for permanent failures\n    - Multiple authentication methods (Bearer, Basic, API Key)\n\n    Use cases:\n    - Critical webhook delivery requiring guaranteed delivery\n    - Integration with unreliable third-party webhooks\n    - High-volume webhook processing with failure protection\n    - Webhook monitoring and alerting\n\n    Process:\n    1. Check circuit breaker state (if enabled)\n    2. Attempt webhook delivery with HTTP bubble\n    3. On failure, calculate backoff delay with jitter\n    4. Retry with exponential backoff until max attempts\n    5. Update circuit breaker state based on results\n    6. Return detailed delivery status\n  ";
    static readonly alias = "webhook-repeat";
    private static circuitBreakerState;
    constructor(params: WebhookRepeaterParams, context?: BubbleContext);
    protected performAction(): Promise<WebhookRepeaterResult>;
    /**
     * Calculate exponential backoff delay with optional jitter
     */
    private calculateBackoffDelay;
    /**
     * Sleep for specified milliseconds
     */
    private sleep;
    /**
     * Generate unique webhook ID
     */
    private generateId;
    /**
     * Reset circuit breaker for a specific webhook URL (utility method)
     */
    static resetCircuitBreaker(webhookUrl: string): void;
    /**
     * Get circuit breaker state for a webhook URL (utility method)
     */
    static getCircuitBreakerState(webhookUrl: string): {
        isOpen: boolean;
        failureCount: number;
        lastFailureTime?: Date;
    } | undefined;
}
export {};
//# sourceMappingURL=webhook-repeater.workflow.d.ts.map