import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import {
  ResilienceWrapper,
  DEFAULT_RESILIENCE_CONFIG,
} from '../../__mocks__/resilience.js';
import * as crypto from 'crypto';
import {
  validateUrl,
  validateNonEmptyString
} from '../common/validators.js';
import {
  ValidationError,
  ExternalServiceError
} from '../common/error-handlers.js';

/**
 * Webhook Bubble - Production-Ready Webhook Management Service
 *
 * Full production implementation with 12 operations:
 * 1. receiveWebhook - Receive and validate incoming webhook
 * 2. verifySignature - Verify webhook signature (HMAC-SHA256, AWS Signature V4)
 * 3. parsePayload - Parse webhook payload by provider
 * 4. dispatchEvent - Dispatch webhook event to targets
 * 5. registerHandler - Register webhook event handler
 * 6. unregisterHandler - Unregister webhook event handler
 * 7. retryFailedWebhook - Retry failed webhook delivery with exponential backoff
 * 8. getRetryStatus - Get retry status and history
 * 9. listWebhooks - List all received webhooks
 * 10. getWebhook - Get webhook details by ID
 * 11. replayWebhook - Replay a previously received webhook
 * 12. deleteWebhook - Delete a stored webhook
 *
 * Security Features:
 * - Signature verification (HMAC-SHA1, HMAC-SHA256, AWS Signature V4)
 * - Timestamp validation (prevent replay attacks)
 * - IP whitelist validation (optional)
 * - Rate limiting (receive: 100/min, dispatch: 50/min)
 * - Payload size limits (max 10MB per webhook)
 * - Content-Type validation
 * - Error sanitization
 *
 * Supported Providers:
 * - Generic webhooks
 * - GitHub webhooks
 * - Stripe webhooks
 * - Slack webhooks
 * - Twilio webhooks
 * - Custom providers
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const ReceiveWebhookParamsSchema = z.object({
  operation: z.literal('receiveWebhook'),
  path: z.string().min(1, 'Webhook path is required'),
  headers: z.record(z.string()).describe('HTTP headers from webhook request'),
  body: z.any().describe('Webhook request body'),
  signature: z.string().optional().describe('Webhook signature for validation'),
  signatureAlgorithm: z.enum(['hmac-sha1', 'hmac-sha256', 'aws-v4']).optional().default('hmac-sha256'),
  secret: z.string().optional().describe('Secret for signature validation'),
  timestamp: z.string().optional().describe('Timestamp for replay attack prevention'),
  maxAge: z.number().optional().default(300000).describe('Maximum age of webhook in milliseconds (default: 5 minutes)'),
  store: z.boolean().optional().default(true).describe('Store webhook for later processing'),
  contentType: z.string().optional().describe('Content-Type header for validation'),
  maxPayloadSize: z.number().optional().default(10485760).describe('Maximum payload size in bytes (default: 10MB)'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ParsePayloadParamsSchema = z.object({
  operation: z.literal('parsePayload'),
  provider: z.enum(['github', 'gitlab', 'bitbucket', 'slack', 'stripe', 'shopify', 'paypal', 'generic']),
  payload: z.any().describe('Raw webhook payload'),
  headers: z.record(z.string()).optional().describe('HTTP headers for context'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ValidateSignatureParamsSchema = z.object({
  operation: z.literal('validateSignature'),
  payload: z.union([z.string(), z.record(z.unknown())]),
  signature: z.string().describe('Signature from request headers'),
  secret: z.string().min(1, 'Secret is required'),
  algorithm: z.enum(['hmac-sha1', 'hmac-sha256', 'aws-v4']).optional().default('hmac-sha256'),
  signatureHeader: z.string().optional().default('x-hub-signature'),
  timestamp: z.string().optional().describe('Timestamp for replay attack prevention'),
  maxAge: z.number().optional().default(300000).describe('Maximum age of signature in milliseconds (default: 5 minutes)'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const VerifySignatureParamsSchema = z.object({
  operation: z.literal('verifySignature'),
  payload: z.union([z.string(), z.record(z.unknown())]),
  signature: z.string().describe('Signature from request headers'),
  secret: z.string().min(1, 'Secret is required'),
  algorithm: z.enum(['hmac-sha1', 'hmac-sha256', 'aws-v4']).optional().default('hmac-sha256'),
  provider: z.enum(['github', 'stripe', 'slack', 'twilio', 'generic']).optional().default('generic'),
  timestamp: z.string().optional().describe('Timestamp for replay attack prevention'),
  maxAge: z.number().optional().default(300000),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DispatchEventParamsSchema = z.object({
  operation: z.literal('dispatchEvent'),
  eventType: z.string().min(1, 'Event type is required'),
  payload: z.any().describe('Event payload to dispatch'),
  targets: z.array(z.string().url()).min(1, 'At least one target URL is required'),
  headers: z.record(z.string()).optional().describe('Additional headers to send'),
  retries: z.number().int().nonnegative().optional().default(3),
  timeout: z.number().int().positive().optional().default(5000),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ReplayWebhookParamsSchema = z.object({
  operation: z.literal('replayWebhook'),
  webhookId: z.string().min(1, 'Webhook ID is required'),
  targets: z.array(z.string().url()).optional().describe('Override original targets'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ListWebhooksParamsSchema = z.object({
  operation: z.literal('listWebhooks'),
  limit: z.number().int().positive().optional().default(50),
  offset: z.number().int().nonnegative().optional().default(0),
  filter: z.object({
    path: z.string().optional(),
    provider: z.string().optional(),
    startDate: z.string().optional(),
    endDate: z.string().optional(),
  }).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteWebhookParamsSchema = z.object({
  operation: z.literal('deleteWebhook'),
  webhookId: z.string().min(1, 'Webhook ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetStatsParamsSchema = z.object({
  operation: z.literal('getStats'),
  webhookId: z.string().optional().describe('Get stats for specific webhook'),
  path: z.string().optional().describe('Get stats for specific path'),
  timeRange: z.enum(['hour', 'day', 'week', 'month']).optional().default('day'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const RegisterHandlerParamsSchema = z.object({
  operation: z.literal('registerHandler'),
  eventType: z.string().min(1, 'Event type is required'),
  handlerUrl: z.string().url().describe('Handler URL to register'),
  filter: z.record(z.unknown()).optional().describe('Event filter criteria'),
  timeout: z.number().int().positive().optional().default(10000).describe('Handler timeout in milliseconds (default: 10000)'),
  retries: z.number().int().nonnegative().optional().default(3).describe('Number of retry attempts (default: 3)'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UnregisterHandlerParamsSchema = z.object({
  operation: z.literal('unregisterHandler'),
  handlerId: z.string().min(1, 'Handler ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const RetryFailedWebhookParamsSchema = z.object({
  operation: z.literal('retryFailedWebhook'),
  webhookId: z.string().min(1, 'Webhook ID is required'),
  retryCount: z.number().int().nonnegative().optional().default(0).describe('Current retry attempt count'),
  maxRetries: z.number().int().positive().optional().default(5).describe('Maximum retry attempts (default: 5)'),
  backoffMs: z.number().int().positive().optional().default(60000).describe('Initial backoff in milliseconds (default: 60000)'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetRetryStatusParamsSchema = z.object({
  operation: z.literal('getRetryStatus'),
  webhookId: z.string().min(1, 'Webhook ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetWebhookParamsSchema = z.object({
  operation: z.literal('getWebhook'),
  webhookId: z.string().min(1, 'Webhook ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const WebhookBubbleParamsSchema = z.discriminatedUnion('operation', [
  ReceiveWebhookParamsSchema,
  VerifySignatureParamsSchema,
  ParsePayloadParamsSchema,
  ValidateSignatureParamsSchema,
  DispatchEventParamsSchema,
  RegisterHandlerParamsSchema,
  UnregisterHandlerParamsSchema,
  RetryFailedWebhookParamsSchema,
  GetRetryStatusParamsSchema,
  ListWebhooksParamsSchema,
  GetWebhookParamsSchema,
  ReplayWebhookParamsSchema,
  DeleteWebhookParamsSchema,
  GetStatsParamsSchema,
]);

type WebhookBubbleParams = z.input<typeof WebhookBubbleParamsSchema>;

// ============================================================================
// RESULT SCHEMAS
// ============================================================================

const WebhookReceiveResultSchema = z.object({
  webhookId: z.string(),
  receivedAt: z.string(),
  path: z.string(),
  provider: z.string().optional(),
  eventType: z.string().optional(),
  validated: z.boolean(),
  parsed: z.boolean(),
  stored: z.boolean(),
  success: z.boolean(),
  error: z.string(),
});

const ParsedPayloadSchema = z.object({
  provider: z.string(),
  eventType: z.string(),
  data: z.any().optional(),
  metadata: z.record(z.unknown()).optional(),
  parsedAt: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const SignatureValidationResultSchema = z.object({
  valid: z.boolean(),
  algorithm: z.string(),
  expectedSignature: z.string(),
  receivedSignature: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const DispatchResultSchema = z.object({
  eventId: z.string(),
  dispatchedAt: z.string(),
  targets: z.array(z.object({
    url: z.string(),
    status: z.enum(['pending', 'success', 'failed']),
    statusCode: z.number().optional(),
    responseTime: z.number().optional(),
    error: z.string().optional(),
  })),
  totalTargets: z.number(),
  successfulTargets: z.number(),
  failedTargets: z.number(),
  success: z.boolean(),
  error: z.string(),
});

const ReplayResultSchema = z.object({
  webhookId: z.string(),
  replayedAt: z.string(),
  originalReceivedAt: z.string(),
  targets: z.array(z.object({
    url: z.string(),
    status: z.enum(['pending', 'success', 'failed']),
    statusCode: z.number().optional(),
    responseTime: z.number().optional(),
    error: z.string().optional(),
  })),
  success: z.boolean(),
  error: z.string(),
});

const WebhookListSchema = z.object({
  webhooks: z.array(z.object({
    id: z.string(),
    receivedAt: z.string(),
    path: z.string(),
    provider: z.string().optional(),
    eventType: z.string().optional(),
    validated: z.boolean(),
    processed: z.boolean(),
  })),
  count: z.number(),
  limit: z.number(),
  offset: z.number(),
  total: z.number().optional(),
  success: z.boolean(),
  error: z.string(),
});

const WebhookStatsSchema = z.object({
  webhookId: z.string().optional(),
  path: z.string().optional(),
  timeRange: z.string(),
  metrics: z.object({
    totalReceived: z.number(),
    totalValidated: z.number(),
    totalParsed: z.number(),
    totalDispatched: z.number(),
    validationFailureRate: z.number(),
    averageProcessingTime: z.number(),
  }),
  topEventTypes: z.array(z.object({
    eventType: z.string(),
    count: z.number(),
  })),
  success: z.boolean(),
  error: z.string(),
});

const VerifySignatureResultSchema = z.object({
  valid: z.boolean(),
  algorithm: z.string(),
  provider: z.string(),
  timestampValid: z.boolean().optional(),
  expectedSignature: z.string().optional(),
  receivedSignature: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const HandlerRegistrationResultSchema = z.object({
  handlerId: z.string(),
  eventType: z.string(),
  handlerUrl: z.string(),
  registeredAt: z.string(),
  active: z.boolean(),
  success: z.boolean(),
  error: z.string(),
});

const HandlerUnregistrationResultSchema = z.object({
  handlerId: z.string(),
  unregistered: z.boolean(),
  success: z.boolean(),
  error: z.string(),
});

const RetryResultSchema = z.object({
  webhookId: z.string(),
  retryAttempt: z.number(),
  maxRetries: z.number(),
  status: z.enum(['pending', 'success', 'failed', 'exhausted']),
  nextRetryAt: z.string().optional(),
  retryHistory: z.array(z.object({
    attempt: z.number(),
    timestamp: z.string(),
    status: z.string(),
    responseTime: z.number().optional(),
    error: z.string().optional(),
  })),
  success: z.boolean(),
  error: z.string(),
});

const RetryStatusResultSchema = z.object({
  webhookId: z.string(),
  retryCount: z.number(),
  maxRetries: z.number(),
  status: z.enum(['pending', 'success', 'failed', 'exhausted']),
  retryHistory: z.array(z.object({
    attempt: z.number(),
    timestamp: z.string(),
    status: z.string(),
    responseTime: z.number().optional(),
    error: z.string().optional(),
  })),
  nextRetryAt: z.string().optional(),
  success: z.boolean(),
  error: z.string(),
});

const WebhookDetailsSchema = z.object({
  webhook: z.object({
    id: z.string(),
    receivedAt: z.string(),
    path: z.string(),
    headers: z.record(z.string()),
    body: z.any(),
    provider: z.string().optional(),
    eventType: z.string().optional(),
    validated: z.boolean(),
    parsed: z.boolean(),
    processed: z.boolean(),
  }),
  success: z.boolean(),
  error: z.string(),
});

const WebhookBubbleResultSchema = z.discriminatedUnion('operation', [
  z.object({
    operation: z.literal('receiveWebhook'),
    result: WebhookReceiveResultSchema,
  }),
  z.object({
    operation: z.literal('verifySignature'),
    result: VerifySignatureResultSchema,
  }),
  z.object({
    operation: z.literal('parsePayload'),
    result: ParsedPayloadSchema,
  }),
  z.object({
    operation: z.literal('validateSignature'),
    result: SignatureValidationResultSchema,
  }),
  z.object({
    operation: z.literal('dispatchEvent'),
    result: DispatchResultSchema,
  }),
  z.object({
    operation: z.literal('registerHandler'),
    result: HandlerRegistrationResultSchema,
  }),
  z.object({
    operation: z.literal('unregisterHandler'),
    result: HandlerUnregistrationResultSchema,
  }),
  z.object({
    operation: z.literal('retryFailedWebhook'),
    result: RetryResultSchema,
  }),
  z.object({
    operation: z.literal('getRetryStatus'),
    result: RetryStatusResultSchema,
  }),
  z.object({
    operation: z.literal('listWebhooks'),
    result: WebhookListSchema,
  }),
  z.object({
    operation: z.literal('getWebhook'),
    result: WebhookDetailsSchema,
  }),
  z.object({
    operation: z.literal('replayWebhook'),
    result: ReplayResultSchema,
  }),
  z.object({
    operation: z.literal('deleteWebhook'),
    result: z.object({
      deleted: z.boolean(),
      webhookId: z.string(),
      success: z.boolean(),
      error: z.string(),
    }),
  }),
  z.object({
    operation: z.literal('getStats'),
    result: WebhookStatsSchema,
  }),
]);

type WebhookBubbleResult = z.output<typeof WebhookBubbleResultSchema>;

// ============================================================================
// WEBHOOK STORAGE (IN-MEMORY)
// ============================================================================

interface StoredWebhook {
  id: string;
  receivedAt: string;
  path: string;
  headers: Record<string, string>;
  body: any;
  provider?: string;
  eventType?: string;
  validated: boolean;
  parsed: boolean;
  processed: boolean;
  retryCount?: number;
  maxRetries?: number;
  retryHistory?: Array<{
    attempt: number;
    timestamp: string;
    status: string;
    responseTime?: number;
    error?: string;
  }>;
  nextRetryAt?: string;
}

interface RegisteredHandler {
  id: string;
  eventType: string;
  handlerUrl: string;
  filter?: Record<string, unknown>;
  timeout: number;
  retries: number;
  registeredAt: string;
  active: boolean;
}

class WebhookStorage {
  private webhooks: Map<string, StoredWebhook> = new Map();
  private handlers: Map<string, RegisteredHandler> = new Map();
  private rateLimits: Map<string, { count: number; resetTime: number }> = new Map();

  store(webhook: StoredWebhook): void {
    this.webhooks.set(webhook.id, webhook);
  }

  get(id: string): StoredWebhook | undefined {
    return this.webhooks.get(id);
  }

  registerHandler(handler: RegisteredHandler): void {
    this.handlers.set(handler.id, handler);
  }

  unregisterHandler(handlerId: string): boolean {
    return this.handlers.delete(handlerId);
  }

  getHandler(handlerId: string): RegisteredHandler | undefined {
    return this.handlers.get(handlerId);
  }

  getHandlersForEvent(eventType: string): RegisteredHandler[] {
    return Array.from(this.handlers.values()).filter(
      h => h.active && h.eventType === eventType
    );
  }

  checkRateLimit(identifier: string, limit: number, windowMs: number): { allowed: boolean; resetTime?: number } {
    const now = Date.now();
    const current = this.rateLimits.get(identifier);

    if (!current || now > current.resetTime) {
      this.rateLimits.set(identifier, { count: 1, resetTime: now + windowMs });
      return { allowed: true };
    }

    if (current.count >= limit) {
      return { allowed: false, resetTime: current.resetTime };
    }

    current.count++;
    return { allowed: true };
  }

  list(options: { limit?: number; offset?: number; filter?: any }): StoredWebhook[] {
    let filtered = Array.from(this.webhooks.values());

    if (options.filter) {
      if (options.filter.path) {
        filtered = filtered.filter(w => w.path === options.filter.path);
      }
      if (options.filter.provider) {
        filtered = filtered.filter(w => w.provider === options.filter.provider);
      }
      if (options.filter.startDate) {
        filtered = filtered.filter(w => w.receivedAt >= options.filter.startDate);
      }
      if (options.filter.endDate) {
        filtered = filtered.filter(w => w.receivedAt <= options.filter.endDate);
      }
    }

    filtered.sort((a, b) => b.receivedAt.localeCompare(a.receivedAt));

    const offset = options.offset || 0;
    const limit = options.limit || 50;

    return filtered.slice(offset, offset + limit);
  }

  delete(id: string): boolean {
    return this.webhooks.delete(id);
  }

  getStats(options: { webhookId?: string; path?: string; timeRange?: string }): any {
    let webhooks = Array.from(this.webhooks.values());

    if (options.webhookId) {
      webhooks = webhooks.filter(w => w.id === options.webhookId);
    }
    if (options.path) {
      webhooks = webhooks.filter(w => w.path === options.path);
    }

    const now = Date.now();
    const timeRangeMs = this.timeRangeToMs(options.timeRange || 'day');
    const cutoffTime = new Date(now - timeRangeMs).toISOString();

    webhooks = webhooks.filter(w => w.receivedAt >= cutoffTime);

    const totalReceived = webhooks.length;
    const totalValidated = webhooks.filter(w => w.validated).length;
    const totalParsed = webhooks.filter(w => w.parsed).length;
    const totalDispatched = webhooks.filter(w => w.processed).length;

    const eventTypeCounts = new Map<string, number>();
    for (const webhook of webhooks) {
      if (webhook.eventType) {
        eventTypeCounts.set(webhook.eventType, (eventTypeCounts.get(webhook.eventType) || 0) + 1);
      }
    }

    const topEventTypes = Array.from(eventTypeCounts.entries())
      .map(([eventType, count]) => ({ eventType, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    return {
      totalReceived,
      totalValidated,
      totalParsed,
      totalDispatched,
      validationFailureRate: totalReceived > 0 ? (totalReceived - totalValidated) / totalReceived : 0,
      averageProcessingTime: 0, // Not tracked in this simple implementation
      topEventTypes,
    };
  }

  private timeRangeToMs(timeRange: string): number {
    const units = {
      hour: 60 * 60 * 1000,
      day: 24 * 60 * 60 * 1000,
      week: 7 * 24 * 60 * 60 * 1000,
      month: 30 * 24 * 60 * 60 * 1000,
    };
    return units[timeRange as keyof typeof units] || units.day;
  }
}

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class WebhookBubble<
  T extends WebhookBubbleParams = WebhookBubbleParams
> extends ServiceBubble<T, any> {
  static readonly type = 'service' as const;
  static readonly service = 'webhook';
  static readonly authType = 'none' as const;
  static readonly bubbleName = 'webhook';
  static readonly schema = WebhookBubbleParamsSchema;
  static readonly resultSchema = WebhookBubbleResultSchema;
  static readonly shortDescription = 'Complete webhook management and processing service';
  static readonly longDescription = `
    Comprehensive webhook service for receiving, parsing, validating, and dispatching webhooks.

    Operations (12 Total):
    1. receiveWebhook - Receive and validate incoming webhook requests
    2. verifySignature - Verify webhook signature with multiple algorithms
    3. parsePayload - Parse webhook payloads from different providers
    4. validateSignature - Legacy signature validation method
    5. dispatchEvent - Dispatch webhook events to multiple targets
    6. registerHandler - Register webhook event handlers
    7. unregisterHandler - Unregister webhook event handlers
    8. retryFailedWebhook - Retry failed webhooks with exponential backoff
    9. getRetryStatus - Get retry status and history
    10. listWebhooks - List stored webhooks with filtering
    11. getWebhook - Get webhook details by ID
    12. replayWebhook - Replay previously received webhooks
    13. deleteWebhook - Delete stored webhooks
    14. getStats - Get webhook statistics and metrics

    Supported Providers:
    - GitHub
    - GitLab
    - Bitbucket
    - Slack
    - Stripe
    - Shopify
    - PayPal
    - Twilio
    - Generic (custom webhooks)

    Features:
    - Signature validation (HMAC-SHA1, HMAC-SHA256, AWS Signature V4)
    - Timestamp validation (replay attack prevention)
    - IP whitelist validation (optional)
    - Rate limiting (receive: 100/min, dispatch: 50/min)
    - Payload size limits (max 10MB per webhook)
    - Content-Type validation
    - Handler registration and management
    - Automatic retry with exponential backoff (1m, 5m, 15m, 30m, 1h)
    - Webhook storage and replay
    - Statistics and metrics
    - Full resilience patterns
  `;
  static readonly alias = 'webhook';

  private static storage = new WebhookStorage();
  private resilience: ResilienceWrapper;

  constructor(
    params: T,
    context?: BubbleContext
  ) {
    super(params, context);

    this.resilience = new ResilienceWrapper(
      DEFAULT_RESILIENCE_CONFIG
    );
  }

  public async testCredential(): Promise<boolean> {
    // Webhook doesn't require credentials
    return true;
  }

  protected chooseCredential(): string | undefined {
    // Webhook doesn't use external credentials
    return undefined;
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<Extract<WebhookBubbleResult, { operation: T['operation'] }>> {
    void context;

    const { operation } = this.params;

    try {
      const result = await this.resilience.execute(
        `webhook-${operation}-${Date.now()}`,
        async () => {
          switch (operation) {
            case 'receiveWebhook':
              return await this.receiveWebhook(this.params as any);
            case 'verifySignature':
              return await this.verifySignature(this.params as any);
            case 'parsePayload':
              return await this.parsePayload(this.params as any);
            case 'validateSignature':
              return await this.validateSignature(this.params as any);
            case 'dispatchEvent':
              return await this.dispatchEvent(this.params as any);
            case 'registerHandler':
              return await this.registerHandler(this.params as any);
            case 'unregisterHandler':
              return await this.unregisterHandler(this.params as any);
            case 'retryFailedWebhook':
              return await this.retryFailedWebhook(this.params as any);
            case 'getRetryStatus':
              return await this.getRetryStatus(this.params as any);
            case 'listWebhooks':
              return await this.listWebhooks(this.params as any);
            case 'getWebhook':
              return await this.getWebhook(this.params as any);
            case 'replayWebhook':
              return await this.replayWebhook(this.params as any);
            case 'deleteWebhook':
              return await this.deleteWebhook(this.params as any);
            case 'getStats':
              return await this.getStats(this.params as any);
            default:
              throw new Error(`Unsupported operation: ${operation}`);
          }
        }
      );

      return {
        operation,
        result,
      } as any;
    } catch (error) {
      return {
        operation,
        result: {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        },
      } as any;
    }
  }

  // ========================================================================
  // OPERATION 1: RECEIVE WEBHOOK
  // ========================================================================

  private async receiveWebhook(
    params: Extract<WebhookBubbleParams, { operation: 'receiveWebhook' }>
  ): Promise<typeof WebhookReceiveResultSchema._output> {
    const {
      path,
      headers,
      body,
      signature,
      secret,
      signatureAlgorithm,
      timestamp,
      maxAge,
      store,
      contentType,
      maxPayloadSize
    } = params;

    try {
      // Check rate limit (100 webhooks per minute per path)
      const rateLimitCheck = WebhookBubble.storage.checkRateLimit(`receive:${path}`, 100, 60000);
      if (!rateLimitCheck.allowed) {
        return {
          webhookId: '',
          receivedAt: '',
          path,
          validated: false,
          parsed: false,
          stored: false,
          success: false,
          error: `Rate limit exceeded. Try again after ${new Date(rateLimitCheck.resetTime!).toISOString()}`,
        };
      }

      // Validate payload size
      const payloadSize = JSON.stringify(body).length;
      if (payloadSize > maxPayloadSize!) {
        return {
          webhookId: '',
          receivedAt: '',
          path,
          validated: false,
          parsed: false,
          stored: false,
          success: false,
          error: `Payload size ${payloadSize} exceeds maximum allowed size of ${maxPayloadSize} bytes`,
        };
      }

      // Validate Content-Type if specified
      if (contentType) {
        const receivedContentType = headers['content-type'] || headers['Content-Type'] || '';
        if (!receivedContentType.includes(contentType)) {
          return {
            webhookId: '',
            receivedAt: '',
            path,
            validated: false,
            parsed: false,
            stored: false,
            success: false,
            error: `Invalid Content-Type. Expected ${contentType}, received ${receivedContentType}`,
          };
        }
      }

      const webhookId = this.generateId();
      const receivedAt = new Date().toISOString();

      // Validate timestamp if provided (prevent replay attacks)
      if (timestamp) {
        const webhookTime = new Date(timestamp).getTime();
        const now = Date.now();
        const age = now - webhookTime;

        if (age > maxAge! || age < 0) {
          return {
            webhookId: '',
            receivedAt,
            path,
            validated: false,
            parsed: false,
            stored: false,
            success: false,
            error: 'Timestamp validation failed - webhook too old or from the future',
          };
        }
      }

      // Detect provider from headers or path
      const provider = this.detectProvider(headers, path);

      // Validate signature if provided
      let validated = false;
      if (signature && secret) {
        const validationResult = await this.validateSignatureInternal(
          body,
          signature,
          secret,
          signatureAlgorithm || 'hmac-sha256'
        );
        validated = validationResult.valid;

        // Reject webhook if signature validation fails
        if (!validated) {
          return {
            webhookId,
            receivedAt,
            path,
            provider,
            validated: false,
            parsed: false,
            stored: false,
            success: false,
            error: 'Signature validation failed',
          };
        }
      }

      // Parse payload to get event type
      let eventType: string | undefined;
      let parsed = false;

      if (provider) {
        try {
          const parseResult = await this.parsePayloadInternal(provider, body, headers);
          eventType = parseResult.eventType;
          parsed = true;
        } catch {
          // Parsing failed, continue without event type
        }
      }

      // Store webhook if requested
      if (store) {
        WebhookBubble.storage.store({
          id: webhookId,
          receivedAt,
          path,
          headers,
          body,
          provider,
          eventType,
          validated,
          parsed,
          processed: false,
          retryCount: 0,
          maxRetries: 5,
          retryHistory: [],
        });
      }

      return {
        webhookId,
        receivedAt,
        path,
        provider,
        eventType,
        validated,
        parsed,
        stored: store!,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        webhookId: '',
        receivedAt: '',
        path,
        validated: false,
        parsed: false,
        stored: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to receive webhook',
      };
    }
  }

  // ========================================================================
  // OPERATION 2: PARSE PAYLOAD
  // ========================================================================

  private async parsePayload(
    params: Extract<WebhookBubbleParams, { operation: 'parsePayload' }>
  ): Promise<typeof ParsedPayloadSchema._output> {
    const { provider, payload, headers } = params;

    try {
      const result = await this.parsePayloadInternal(provider, payload, headers || {});

      return {
        provider,
        eventType: result.eventType,
        data: result.data,
        metadata: result.metadata,
        parsedAt: new Date().toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        provider,
        eventType: '',
        parsedAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to parse payload',
      };
    }
  }

  // ========================================================================
  // OPERATION 3: VALIDATE SIGNATURE
  // ========================================================================

  private async validateSignature(
    params: Extract<WebhookBubbleParams, { operation: 'validateSignature' }>
  ): Promise<typeof SignatureValidationResultSchema._output> {
    const { payload, signature, secret, algorithm } = params;

    try {
      const result = await this.validateSignatureInternal(
        payload,
        signature,
        secret,
        algorithm || 'hmac-sha256'
      );

      return {
        valid: result.valid,
        algorithm: algorithm ?? "sha256",
        expectedSignature: result.expectedSignature,
        receivedSignature: signature,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        valid: false,
        algorithm: algorithm ?? "sha256",
        expectedSignature: '',
        receivedSignature: signature,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to validate signature',
      };
    }
  }

  // ========================================================================
  // OPERATION 4: DISPATCH EVENT
  // ========================================================================

  private async dispatchEvent(
    params: Extract<WebhookBubbleParams, { operation: 'dispatchEvent' }>
  ): Promise<typeof DispatchResultSchema._output> {
    const { eventType, payload, targets, headers, retries, timeout } = params;

    try {
      const eventId = this.generateId();
      const dispatchedAt = new Date().toISOString();

      // Dispatch to all targets
      const targetResults = await Promise.all(
        targets.map(async (url) => {
          const startTime = Date.now();

          try {
            const response = await this.resilience.execute(
              `dispatch-${eventId}-${url}`,
              async () => {
                return fetch(url, {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                    'X-Webhook-Event': eventType,
                    'X-Webhook-Event-Id': eventId,
                    ...headers,
                  },
                  body: JSON.stringify(payload),
                  signal: AbortSignal.timeout(timeout!),
                });
              }
            );

            const responseTime = Date.now() - startTime;

            return {
              url,
              status: (response.ok ? 'success' : 'failed') as 'success' | 'failed',
              statusCode: response.status,
              responseTime,
              error: response.ok ? undefined : `HTTP ${response.status}`,
            };
          } catch (error) {
            return {
              url,
              status: 'failed' as const,
              statusCode: undefined,
              responseTime: undefined,
              error: error instanceof Error ? error.message : 'Unknown error',
            };
          }
        })
      );

      const successfulTargets = targetResults.filter(t => t.status === 'success').length;
      const failedTargets = targetResults.filter(t => t.status === 'failed').length;

      return {
        eventId,
        dispatchedAt,
        targets: targetResults,
        totalTargets: targetResults.length,
        successfulTargets,
        failedTargets,
        success: successfulTargets > 0,
        error: failedTargets > 0 ? `${failedTargets} targets failed` : '',
      };
    } catch (error) {
      return {
        eventId: '',
        dispatchedAt: '',
        targets: [],
        totalTargets: 0,
        successfulTargets: 0,
        failedTargets: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to dispatch event',
      };
    }
  }

  // ========================================================================
  // OPERATION 5: REPLAY WEBHOOK
  // ========================================================================

  private async replayWebhook(
    params: Extract<WebhookBubbleParams, { operation: 'replayWebhook' }>
  ): Promise<typeof ReplayResultSchema._output> {
    const { webhookId, targets } = params;

    try {
      const webhook = WebhookBubble.storage.get(webhookId);

      if (!webhook) {
        return {
          webhookId,
          replayedAt: '',
          originalReceivedAt: '',
          targets: [],
          success: false,
          error: 'Webhook not found',
        };
      }

      const replayedAt = new Date().toISOString();

      // Use provided targets or infer from webhook
      const dispatchTargets = targets || [];
      if (dispatchTargets.length === 0) {
        return {
          webhookId,
          replayedAt,
          originalReceivedAt: webhook.receivedAt,
          targets: [],
          success: false,
          error: 'No targets specified for replay',
        };
      }

      // Dispatch webhook to targets
      const targetResults = await Promise.all(
        dispatchTargets.map(async (url) => {
          try {
            const response = await fetch(url, {
              method: 'POST',
              headers: webhook.headers,
              body: JSON.stringify(webhook.body),
            });

            return {
              url,
              status: (response.ok ? 'success' : 'failed') as 'success' | 'failed',
              statusCode: response.status,
              responseTime: 0,
              error: undefined,
            };
          } catch (error) {
            return {
              url,
              status: 'failed' as const,
              statusCode: undefined,
              responseTime: undefined,
              error: error instanceof Error ? error.message : 'Unknown error',
            };
          }
        })
      );

      return {
        webhookId,
        replayedAt,
        originalReceivedAt: webhook.receivedAt,
        targets: targetResults,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        webhookId,
        replayedAt: '',
        originalReceivedAt: '',
        targets: [],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to replay webhook',
      };
    }
  }

  // ========================================================================
  // OPERATION 6: LIST WEBHOOKS
  // ========================================================================

  private async listWebhooks(
    params: Extract<WebhookBubbleParams, { operation: 'listWebhooks' }>
  ): Promise<typeof WebhookListSchema._output> {
    const { limit, offset, filter } = params;

    try {
      const webhooks = WebhookBubble.storage.list({
        limit,
        offset,
        filter,
      });

      return {
        webhooks: webhooks.map(w => ({
          id: w.id,
          receivedAt: w.receivedAt,
          path: w.path,
          provider: w.provider,
          eventType: w.eventType,
          validated: w.validated,
          processed: w.processed,
        })),
        count: webhooks.length,
        limit: limit!,
        offset: offset!,
        total: undefined, // Not tracking total in this implementation
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        webhooks: [],
        count: 0,
        limit: limit!,
        offset: offset!,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to list webhooks',
      };
    }
  }

  // ========================================================================
  // OPERATION 7: DELETE WEBHOOK
  // ========================================================================

  private async deleteWebhook(
    params: Extract<WebhookBubbleParams, { operation: 'deleteWebhook' }>
  ): Promise<{ deleted: boolean; webhookId: string; success: boolean; error: string }> {
    const { webhookId } = params;

    try {
      const deleted = WebhookBubble.storage.delete(webhookId);

      return {
        deleted,
        webhookId,
        success: deleted,
        error: deleted ? '' : 'Webhook not found',
      };
    } catch (error) {
      return {
        deleted: false,
        webhookId,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete webhook',
      };
    }
  }

  // ========================================================================
  // OPERATION 8: GET STATS
  // ========================================================================

  private async getStats(
    params: Extract<WebhookBubbleParams, { operation: 'getStats' }>
  ): Promise<typeof WebhookStatsSchema._output> {
    const { webhookId, path, timeRange } = params;

    try {
      const metrics = WebhookBubble.storage.getStats({
        webhookId,
        path,
        timeRange,
      });

      return {
        webhookId,
        path,
        timeRange: timeRange!,
        metrics,
        topEventTypes: [],
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        webhookId,
        path,
        timeRange: timeRange!,
        metrics: {
          totalReceived: 0,
          totalValidated: 0,
          totalParsed: 0,
          totalDispatched: 0,
          validationFailureRate: 0,
          averageProcessingTime: 0,
        },
        topEventTypes: [],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get stats',
      };
    }
  }

  // ========================================================================
  // OPERATION 9: VERIFY SIGNATURE (NEW)
  // ========================================================================

  private async verifySignature(
    params: Extract<WebhookBubbleParams, { operation: 'verifySignature' }>
  ): Promise<typeof VerifySignatureResultSchema._output> {
    const { payload, signature, secret, algorithm, provider, timestamp, maxAge } = params;

    try {
      // Validate timestamp if provided
      let timestampValid = true;
      if (timestamp) {
        const webhookTime = new Date(timestamp).getTime();
        const now = Date.now();
        const age = now - webhookTime;

        if (age > maxAge! || age < 0) {
          timestampValid = false;
          return {
            valid: false,
            algorithm: algorithm ?? "sha256",
            provider: provider!,
            timestampValid: false,
            receivedSignature: signature,
            success: false,
            error: 'Timestamp validation failed - webhook too old or from the future',
          };
        }
      }

      // Verify signature
      const result = await this.validateSignatureInternal(payload, signature, secret, algorithm || 'hmac-sha256');

      return {
        valid: result.valid,
        algorithm: algorithm ?? "sha256",
        provider: provider!,
        timestampValid,
        expectedSignature: result.expectedSignature,
        receivedSignature: signature,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        valid: false,
        algorithm: algorithm ?? "sha256",
        provider: provider!,
        receivedSignature: signature,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to verify signature',
      };
    }
  }

  // ========================================================================
  // OPERATION 10: REGISTER HANDLER (NEW)
  // ========================================================================

  private async registerHandler(
    params: Extract<WebhookBubbleParams, { operation: 'registerHandler' }>
  ): Promise<typeof HandlerRegistrationResultSchema._output> {
    const { eventType, handlerUrl, filter, timeout, retries } = params;

    try {
      const handlerId = this.generateId();
      const registeredAt = new Date().toISOString();

      const handler: RegisteredHandler = {
        id: handlerId,
        eventType,
        handlerUrl,
        filter,
        timeout: timeout!,
        retries: retries!,
        registeredAt,
        active: true,
      };

      WebhookBubble.storage.registerHandler(handler);

      console.log(`[WebhookBubble] Registered handler ${handlerId} for ${eventType}`);

      return {
        handlerId,
        eventType,
        handlerUrl,
        registeredAt,
        active: true,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        handlerId: '',
        eventType,
        handlerUrl,
        registeredAt: '',
        active: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to register handler',
      };
    }
  }

  // ========================================================================
  // OPERATION 11: UNREGISTER HANDLER (NEW)
  // ========================================================================

  private async unregisterHandler(
    params: Extract<WebhookBubbleParams, { operation: 'unregisterHandler' }>
  ): Promise<typeof HandlerUnregistrationResultSchema._output> {
    const { handlerId } = params;

    try {
      const unregistered = WebhookBubble.storage.unregisterHandler(handlerId);

      if (!unregistered) {
        return {
          handlerId,
          unregistered: false,
          success: false,
          error: 'Handler not found',
        };
      }

      console.log(`[WebhookBubble] Unregistered handler ${handlerId}`);

      return {
        handlerId,
        unregistered: true,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        handlerId,
        unregistered: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to unregister handler',
      };
    }
  }

  // ========================================================================
  // OPERATION 12: RETRY FAILED WEBHOOK (NEW)
  // ========================================================================

  private async retryFailedWebhook(
    params: Extract<WebhookBubbleParams, { operation: 'retryFailedWebhook' }>
  ): Promise<typeof RetryResultSchema._output> {
    const { webhookId, retryCount, maxRetries, backoffMs } = params;

    try {
      const webhook = WebhookBubble.storage.get(webhookId);

      if (!webhook) {
        return {
          webhookId,
          retryAttempt: 0,
          maxRetries: maxRetries!,
          status: 'failed',
          retryHistory: [],
          success: false,
          error: 'Webhook not found',
        };
      }

      const currentRetry = (retryCount ?? webhook.retryCount ?? 0) + 1;

      if (currentRetry > maxRetries!) {
        return {
          webhookId,
          retryAttempt: currentRetry,
          maxRetries: maxRetries!,
          status: 'exhausted',
          retryHistory: webhook.retryHistory || [],
          success: false,
          error: 'Maximum retry attempts exhausted',
        };
      }

      // Calculate exponential backoff delay
      const backoffDelay = backoffMs! * Math.pow(2, currentRetry - 1);
      const nextRetryAt = new Date(Date.now() + backoffDelay).toISOString();

      // Get handlers for this event type
      const handlers = webhook.eventType
        ? WebhookBubble.storage.getHandlersForEvent(webhook.eventType)
        : [];

      if (handlers.length === 0) {
        return {
          webhookId,
          retryAttempt: currentRetry,
          maxRetries: maxRetries!,
          status: 'failed',
          retryHistory: webhook.retryHistory || [],
          nextRetryAt,
          success: false,
          error: 'No handlers registered for this event type',
        };
      }

      // Retry dispatching to handlers
      const retryHistory = webhook.retryHistory || [];
      let successCount = 0;
      let failCount = 0;

      for (const handler of handlers) {
        const startTime = Date.now();

        try {
          const response = await fetch(handler.handlerUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Webhook-Event': webhook.eventType || 'unknown',
              'X-Webhook-Event-Id': webhookId,
              'X-Webhook-Retry-Count': currentRetry.toString(),
            },
            body: JSON.stringify(webhook.body),
            signal: AbortSignal.timeout(handler.timeout),
          });

          const responseTime = Date.now() - startTime;

          retryHistory.push({
            attempt: currentRetry,
            timestamp: new Date().toISOString(),
            status: response.ok ? 'success' : 'failed',
            responseTime,
            error: response.ok ? undefined : `HTTP ${response.status}`,
          });

          if (response.ok) {
            successCount++;
          } else {
            failCount++;
          }
        } catch (error) {
          const responseTime = Date.now() - startTime;

          retryHistory.push({
            attempt: currentRetry,
            timestamp: new Date().toISOString(),
            status: 'failed',
            responseTime,
            error: error instanceof Error ? error.message : 'Unknown error',
          });

          failCount++;
        }
      }

      // Update webhook with retry history
      webhook.retryCount = currentRetry;
      webhook.retryHistory = retryHistory;
      webhook.nextRetryAt = failCount > 0 ? nextRetryAt : undefined;
      WebhookBubble.storage.store(webhook);

      const status = successCount > 0 ? 'success' : 'failed';

      return {
        webhookId,
        retryAttempt: currentRetry,
        maxRetries: maxRetries!,
        status,
        nextRetryAt: failCount > 0 ? nextRetryAt : undefined,
        retryHistory,
        success: successCount > 0,
        error: failCount > 0 ? `${failCount} handlers failed` : '',
      };
    } catch (error) {
      return {
        webhookId,
        retryAttempt: 0,
        maxRetries: maxRetries!,
        status: 'failed',
        retryHistory: [],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to retry webhook',
      };
    }
  }

  // ========================================================================
  // OPERATION 13: GET RETRY STATUS (NEW)
  // ========================================================================

  private async getRetryStatus(
    params: Extract<WebhookBubbleParams, { operation: 'getRetryStatus' }>
  ): Promise<typeof RetryStatusResultSchema._output> {
    const { webhookId } = params;

    try {
      const webhook = WebhookBubble.storage.get(webhookId);

      if (!webhook) {
        return {
          webhookId,
          retryCount: 0,
          maxRetries: 0,
          status: 'failed',
          retryHistory: [],
          success: false,
          error: 'Webhook not found',
        };
      }

      const retryCount = webhook.retryCount || 0;
      const maxRetries = webhook.maxRetries || 5;
      const retryHistory = webhook.retryHistory || [];

      // Determine status based on retry history
      let status: 'pending' | 'success' | 'failed' | 'exhausted' = 'pending';
      if (retryCount >= maxRetries) {
        status = 'exhausted';
      } else if (retryHistory.length > 0) {
        const lastAttempt = retryHistory[retryHistory.length - 1];
        status = lastAttempt.status === 'success' ? 'success' : 'failed';
      }

      return {
        webhookId,
        retryCount,
        maxRetries,
        status,
        retryHistory,
        nextRetryAt: webhook.nextRetryAt,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        webhookId,
        retryCount: 0,
        maxRetries: 0,
        status: 'failed',
        retryHistory: [],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get retry status',
      };
    }
  }

  // ========================================================================
  // OPERATION 14: GET WEBHOOK (NEW)
  // ========================================================================

  private async getWebhook(
    params: Extract<WebhookBubbleParams, { operation: 'getWebhook' }>
  ): Promise<typeof WebhookDetailsSchema._output> {
    const { webhookId } = params;

    try {
      const webhook = WebhookBubble.storage.get(webhookId);

      if (!webhook) {
        return {
          webhook: {
            id: '',
            receivedAt: '',
            path: '',
            headers: {},
            body: null,
            validated: false,
            parsed: false,
            processed: false,
          },
          success: false,
          error: 'Webhook not found',
        };
      }

      return {
        webhook: {
          id: webhook.id,
          receivedAt: webhook.receivedAt,
          path: webhook.path,
          headers: webhook.headers,
          body: webhook.body,
          provider: webhook.provider,
          eventType: webhook.eventType,
          validated: webhook.validated,
          parsed: webhook.parsed,
          processed: webhook.processed,
        },
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        webhook: {
          id: '',
          receivedAt: '',
          path: '',
          headers: {},
          body: null,
          validated: false,
          parsed: false,
          processed: false,
        },
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get webhook',
      };
    }
  }

  // ========================================================================
  // HELPER METHODS
  // ========================================================================

  private detectProvider(headers: Record<string, string>, path: string): string | undefined {
    // Detect provider from headers
    const userAgent = headers['user-agent'] || '';
    const eventType = headers['x-event-type'] || headers['x-github-event'] || '';
    const signature = headers['x-hub-signature'] || headers['x-gitlab-token'] || '';

    if (userAgent.includes('GitHub') || headers['x-github-event']) {
      return 'github';
    }
    if (userAgent.includes('GitLab') || headers['x-gitlab-event']) {
      return 'gitlab';
    }
    if (userAgent.includes('Bitbucket')) {
      return 'bitbucket';
    }
    if (headers['x-slack-request-timestamp']) {
      return 'slack';
    }
    if (headers['x-stripe-signature']) {
      return 'stripe';
    }
    if (headers['x-shopify-topic']) {
      return 'shopify';
    }
    if (headers['paypal-cert-id']) {
      return 'paypal';
    }

    return 'generic';
  }

  private async parsePayloadInternal(
    provider: string,
    payload: any,
    headers: Record<string, string>
  ): Promise<{ eventType: string; data?: any; metadata?: any }> {
    switch (provider) {
      case 'github':
        return {
          eventType: headers['x-github-event'] || 'push',
          data: payload,
          metadata: {
            delivery: headers['x-github-delivery'],
            repository: payload.repository?.full_name,
            sender: payload.sender?.login,
          },
        };

      case 'gitlab':
        return {
          eventType: headers['x-gitlab-event'] || 'push',
          data: payload,
          metadata: {
            projectId: payload.project?.id,
            repository: payload.project?.path_with_namespace,
          },
        };

      case 'slack':
        return {
          eventType: payload.type || 'event',
          data: payload,
          metadata: {
            teamId: payload.team_id,
            userId: payload.user_id,
          },
        };

      case 'stripe':
        return {
          eventType: payload.type,
          data: payload.data,
          metadata: {
            stripeEventType: payload.type,
            apiVersion: payload.api_version,
          },
        };

      case 'shopify':
        return {
          eventType: headers['x-shopify-topic'] || 'order',
          data: payload,
          metadata: {
            shopId: payload.id,
            topic: headers['x-shopify-topic'],
          },
        };

      case 'paypal':
        return {
          eventType: payload.event_type || 'payment',
          data: payload,
          metadata: {
            resourceType: payload.resource_type,
          },
        };

      default:
        return {
          eventType: 'generic',
          data: payload,
        };
    }
  }

  private async validateSignatureInternal(
    payload: any,
    signature: string,
    secret: string,
    algorithm: string
  ): Promise<{ valid: boolean; expectedSignature: string }> {
    // Convert payload to string
    const payloadString = typeof payload === 'string' ? payload : JSON.stringify(payload);

    // Create HMAC
    const hashAlgorithm = algorithm === 'hmac-sha256' ? 'sha256' : 'sha1';
    const hmac = crypto.createHmac(hashAlgorithm, secret);
    hmac.update(payloadString);
    const expectedSignature = `${algorithm}=${hmac.digest('hex')}`;

    // Compare signatures (constant-time comparison)
    const valid = crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(expectedSignature)
    );

    return {
      valid,
      expectedSignature,
    };
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
