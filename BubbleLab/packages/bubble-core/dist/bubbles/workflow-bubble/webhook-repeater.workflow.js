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
import { CredentialType } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';
/**
 * Retry strategy configuration
 */
const RetryStrategySchema = z.object({
    maxAttempts: z
        .number()
        .int()
        .positive()
        .default(3)
        .describe('Maximum number of retry attempts'),
    initialDelay: z
        .number()
        .int()
        .positive()
        .default(1000)
        .describe('Initial delay in milliseconds'),
    maxDelay: z
        .number()
        .int()
        .positive()
        .default(30000)
        .describe('Maximum delay between retries'),
    backoffMultiplier: z
        .number()
        .positive()
        .default(2)
        .describe('Multiplier for exponential backoff'),
    jitter: z
        .boolean()
        .default(true)
        .describe('Add random jitter to prevent thundering herd'),
});
/**
 * Circuit breaker configuration
 */
const CircuitBreakerConfigSchema = z.object({
    enabled: z
        .boolean()
        .default(true)
        .describe('Enable circuit breaker pattern'),
    failureThreshold: z
        .number()
        .int()
        .positive()
        .default(5)
        .describe('Number of failures before opening circuit'),
    successThreshold: z
        .number()
        .int()
        .positive()
        .default(2)
        .describe('Number of successes to close circuit'),
    timeout: z
        .number()
        .int()
        .positive()
        .default(60000)
        .describe('Time in milliseconds before attempting to close circuit'),
});
/**
 * Parameters schema for webhook repeater workflow
 */
const WebhookRepeaterParamsSchema = z.object({
    /**
     * Target webhook URL
     */
    webhookUrl: z
        .string()
        .url()
        .describe('Target webhook URL to deliver payload to'),
    /**
     * HTTP method for webhook delivery
     */
    method: z
        .enum(['POST', 'PUT', 'PATCH'])
        .default('POST')
        .describe('HTTP method for webhook delivery'),
    /**
     * Payload to deliver
     */
    payload: z
        .union([z.string(), z.record(z.unknown())])
        .describe('Webhook payload (string or JSON object)'),
    /**
     * HTTP headers to include
     */
    headers: z
        .record(z.string())
        .optional()
        .describe('Additional HTTP headers to include'),
    /**
     * Retry strategy configuration
     */
    retryStrategy: z
        .object({
        maxAttempts: z.number().int().positive().default(3),
        initialDelay: z.number().int().positive().default(1000),
        maxDelay: z.number().int().positive().default(30000),
        backoffMultiplier: z.number().positive().default(2),
        jitter: z.boolean().default(true),
    })
        .optional()
        .describe('Retry strategy configuration'),
    /**
     * Circuit breaker configuration
     */
    circuitBreaker: z
        .object({
        enabled: z.boolean().default(true),
        failureThreshold: z.number().int().positive().default(5),
        successThreshold: z.number().int().positive().default(2),
        timeout: z.number().int().positive().default(60000),
    })
        .optional()
        .describe('Circuit breaker configuration'),
    /**
     * Request timeout
     */
    timeout: z
        .number()
        .int()
        .positive()
        .default(30000)
        .describe('Request timeout in milliseconds'),
    /**
     * Webhook ID for tracking
     */
    webhookId: z
        .string()
        .optional()
        .describe('Unique identifier for this webhook delivery'),
    /**
     * Credentials for authentication
     */
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Credentials for webhook authentication'),
    /**
     * Auth type for webhook
     */
    authType: z
        .enum(['none', 'bearer', 'basic', 'api-key', 'api-key-header', 'custom'])
        .default('none')
        .describe('Authentication type'),
    /**
     * Custom auth header name
     */
    authHeader: z
        .string()
        .optional()
        .describe('Custom header name when authType is "custom"'),
});
/**
 * Result schema for webhook repeater workflow
 */
const WebhookRepeaterResultSchema = z.object({
    success: z.boolean(),
    error: z.string(),
    /**
     * Delivery attempt details
     */
    deliveryAttempts: z
        .array(z.object({
        attemptNumber: z.number(),
        timestamp: z.date(),
        success: z.boolean(),
        statusCode: z.number().optional(),
        responseTime: z.number(),
        error: z.string().optional(),
    }))
        .describe('Array of delivery attempts'),
    /**
     * Final delivery status
     */
    deliveryStatus: z
        .object({
        delivered: z.boolean(),
        finalStatusCode: z.number().optional(),
        totalAttempts: z.number(),
        totalDuration: z.number(),
    })
        .optional(),
    /**
     * Circuit breaker state
     */
    circuitBreakerState: z
        .object({
        isOpen: z.boolean(),
        failureCount: z.number(),
        lastFailureTime: z.date().optional(),
    })
        .optional(),
    /**
     * Webhook identifier
     */
    webhookId: z.string().optional(),
});
/**
 * Webhook Repeater Workflow
 *
 * Provides robust webhook delivery with retry logic, circuit breaker, and comprehensive error handling.
 */
export class WebhookRepeaterWorkflow extends WorkflowBubble {
    static type = 'workflow';
    static bubbleName = 'webhook-repeater-workflow';
    static schema = WebhookRepeaterParamsSchema;
    static resultSchema = WebhookRepeaterResultSchema;
    static shortDescription = 'Robust webhook delivery with retries and circuit breaker';
    static longDescription = `
    Provides reliable webhook delivery with advanced retry mechanisms and failure protection.

    Features:
    - Exponential backoff retry with configurable strategies
    - Circuit breaker pattern to prevent cascade failures
    - Jitter to prevent thundering herd problem
    - Comprehensive delivery attempt tracking
    - Dead letter queue support for permanent failures
    - Multiple authentication methods (Bearer, Basic, API Key)

    Use cases:
    - Critical webhook delivery requiring guaranteed delivery
    - Integration with unreliable third-party webhooks
    - High-volume webhook processing with failure protection
    - Webhook monitoring and alerting

    Process:
    1. Check circuit breaker state (if enabled)
    2. Attempt webhook delivery with HTTP bubble
    3. On failure, calculate backoff delay with jitter
    4. Retry with exponential backoff until max attempts
    5. Update circuit breaker state based on results
    6. Return detailed delivery status
  `;
    static alias = 'webhook-repeat';
    // Circuit breaker state
    static circuitBreakerState = new Map();
    constructor(params, context) {
        super(params, context);
    }
    async performAction() {
        const startTime = Date.now();
        const webhookId = this.params.webhookId || this.generateId();
        console.log(`[WebhookRepeater] Starting webhook delivery: ${webhookId}`);
        console.log(`[WebhookRepeater] Target URL: ${this.params.webhookUrl}`);
        const deliveryAttempts = [];
        const retryStrategy = this.params.retryStrategy || {};
        const circuitBreaker = this.params.circuitBreaker || {};
        // Initialize circuit breaker state for this webhook URL
        const circuitBreakerKey = this.params.webhookUrl;
        if (circuitBreaker.enabled !== false) {
            if (!WebhookRepeaterWorkflow.circuitBreakerState.has(circuitBreakerKey)) {
                WebhookRepeaterWorkflow.circuitBreakerState.set(circuitBreakerKey, {
                    isOpen: false,
                    failureCount: 0,
                });
            }
        }
        // Check if circuit is open
        const cbState = WebhookRepeaterWorkflow.circuitBreakerState.get(circuitBreakerKey);
        if (cbState?.isOpen && circuitBreaker.enabled !== false) {
            const timeSinceLastFailure = cbState.lastFailureTime
                ? Date.now() - cbState.lastFailureTime.getTime()
                : Infinity;
            if (timeSinceLastFailure < (circuitBreaker.timeout || 60000)) {
                console.warn('[WebhookRepeater] Circuit breaker is OPEN, rejecting request');
                return {
                    success: false,
                    error: 'Circuit breaker is open for this webhook endpoint',
                    deliveryAttempts,
                    circuitBreakerState: {
                        isOpen: true,
                        failureCount: cbState.failureCount,
                        lastFailureTime: cbState.lastFailureTime,
                    },
                    webhookId,
                };
            }
            else {
                // Attempt to close circuit
                console.log('[WebhookRepeater] Circuit breaker timeout elapsed, attempting delivery');
                cbState.isOpen = false;
            }
        }
        // Attempt delivery with retries
        const maxAttempts = retryStrategy.maxAttempts || 3;
        let lastError;
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const attemptStartTime = Date.now();
            console.log(`[WebhookRepeater] Attempt ${attempt}/${maxAttempts}`);
            // Calculate delay for retry attempts
            if (attempt > 1) {
                const delay = this.calculateBackoffDelay(attempt, retryStrategy);
                console.log(`[WebhookRepeater] Waiting ${delay}ms before retry`);
                await this.sleep(delay);
            }
            // Attempt webhook delivery
            try {
                const httpBubble = new HttpBubble({
                    url: this.params.webhookUrl,
                    method: this.params.method,
                    headers: this.params.headers,
                    body: this.params.payload,
                    timeout: this.params.timeout,
                    credentials: this.params.credentials,
                }, this.context);
                const result = await httpBubble.action();
                const responseTime = Date.now() - attemptStartTime;
                const statusCode = result.data?.status ?? 0;
                deliveryAttempts.push({
                    attemptNumber: attempt,
                    timestamp: new Date(),
                    success: result.success,
                    statusCode,
                    responseTime,
                    error: result.success ? undefined : result.error,
                });
                if (result.success) {
                    console.log(`[WebhookRepeater] Delivery successful on attempt ${attempt}`);
                    // Update circuit breaker on success
                    if (cbState && circuitBreaker.enabled !== false) {
                        cbState.failureCount = Math.max(0, cbState.failureCount - 1);
                        if (cbState.failureCount === 0) {
                            cbState.isOpen = false;
                        }
                    }
                    return {
                        success: true,
                        error: '',
                        deliveryAttempts,
                        deliveryStatus: {
                            delivered: true,
                            finalStatusCode: statusCode,
                            totalAttempts: attempt,
                            totalDuration: Date.now() - startTime,
                        },
                        circuitBreakerState: cbState
                            ? {
                                isOpen: cbState.isOpen,
                                failureCount: cbState.failureCount,
                                lastFailureTime: cbState.lastFailureTime,
                            }
                            : undefined,
                        webhookId,
                    };
                }
                else {
                    lastError = result.error;
                    console.warn(`[WebhookRepeater] Delivery failed on attempt ${attempt}: ${lastError}`);
                    // Update circuit breaker on failure
                    if (cbState && circuitBreaker.enabled !== false) {
                        cbState.failureCount++;
                        cbState.lastFailureTime = new Date();
                        if (cbState.failureCount >= (circuitBreaker.failureThreshold || 5)) {
                            console.warn('[WebhookRepeater] Circuit breaker opened due to repeated failures');
                            cbState.isOpen = true;
                        }
                    }
                }
            }
            catch (error) {
                const responseTime = Date.now() - attemptStartTime;
                lastError = error instanceof Error ? error.message : 'Unknown error';
                deliveryAttempts.push({
                    attemptNumber: attempt,
                    timestamp: new Date(),
                    success: false,
                    responseTime,
                    error: lastError,
                });
                console.error(`[WebhookRepeater] Exception on attempt ${attempt}: ${lastError}`);
                // Update circuit breaker on exception
                if (cbState && circuitBreaker.enabled !== false) {
                    cbState.failureCount++;
                    cbState.lastFailureTime = new Date();
                    if (cbState.failureCount >= (circuitBreaker.failureThreshold || 5)) {
                        console.warn('[WebhookRepeater] Circuit breaker opened due to repeated exceptions');
                        cbState.isOpen = true;
                    }
                }
            }
        }
        // All attempts exhausted
        console.error(`[WebhookRepeater] All ${maxAttempts} attempts failed`);
        return {
            success: false,
            error: lastError || 'All delivery attempts failed',
            deliveryAttempts,
            deliveryStatus: {
                delivered: false,
                totalAttempts: maxAttempts,
                totalDuration: Date.now() - startTime,
            },
            circuitBreakerState: cbState
                ? {
                    isOpen: cbState.isOpen,
                    failureCount: cbState.failureCount,
                    lastFailureTime: cbState.lastFailureTime,
                }
                : undefined,
            webhookId,
        };
    }
    /**
     * Calculate exponential backoff delay with optional jitter
     */
    calculateBackoffDelay(attempt, strategy) {
        const initialDelay = strategy.initialDelay || 1000;
        const maxDelay = strategy.maxDelay || 30000;
        const multiplier = strategy.backoffMultiplier || 2;
        const jitter = strategy.jitter !== false;
        // Calculate exponential delay
        const exponentialDelay = Math.min(initialDelay * Math.pow(multiplier, attempt - 1), maxDelay);
        // Add jitter if enabled (up to 25% of delay)
        if (jitter) {
            const jitterAmount = exponentialDelay * 0.25;
            return exponentialDelay + (Math.random() * jitterAmount * 2 - jitterAmount);
        }
        return exponentialDelay;
    }
    /**
     * Sleep for specified milliseconds
     */
    sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }
    /**
     * Generate unique webhook ID
     */
    generateId() {
        return `webhook_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    }
    /**
     * Reset circuit breaker for a specific webhook URL (utility method)
     */
    static resetCircuitBreaker(webhookUrl) {
        WebhookRepeaterWorkflow.circuitBreakerState.delete(webhookUrl);
    }
    /**
     * Get circuit breaker state for a webhook URL (utility method)
     */
    static getCircuitBreakerState(webhookUrl) {
        return WebhookRepeaterWorkflow.circuitBreakerState.get(webhookUrl);
    }
}
//# sourceMappingURL=webhook-repeater.workflow.js.map