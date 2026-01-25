/**
 * SendGrid API Service Bubble
 *
 * Provides integration with SendGrid API for email operations.
 * Supports sending, templates, lists, scheduling, and analytics.
 *
 * Federation Constitution Compliant
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// SENDGRID-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const SendGridOperationSchema = z.enum([
  'send_email',
  'send_bulk_email',
  'add_recipient_to_list',
  'create_list',
  'get_templates',
  'send_with_template',
  'validate_email',
  'schedule_email',
  'cancel_scheduled_email',
  'get_stats',
  'get_bounces',
  'get_delivery_status',
]);

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const SendGridParamsSchema = z.object({
  operation: SendGridOperationSchema.describe('SendGrid API operation'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  apiKey: z.string().min(1).describe('SendGrid API key (REQUIRED)'),
  baseUrl: z.string().url().default('https://api.sendgrid.com/v3').describe('SendGrid API base URL'),

  // Email operations
  to: z.union([z.string(), z.array(z.string())]).optional().describe('Recipient email address(es)'),
  from: z.string().optional().describe('Sender email address'),
  subject: z.string().optional().describe('Email subject'),
  text: z.string().optional().describe('Plain text email body'),
  html: z.string().optional().describe('HTML email body'),
  templateId: z.string().optional().describe('SendGrid template ID'),
  dynamicData: z.record(z.unknown()).optional().describe('Dynamic template data'),

  // Attachments
  attachments: z.array(z.object({
    content: z.string().describe('Base64 encoded content'),
    filename: z.string().describe('Attachment filename'),
    type: z.string().optional().describe('MIME type'),
    disposition: z.string().optional().describe('Content disposition'),
  })).optional().describe('Email attachments'),

  // Categories and custom args
  categories: z.array(z.string()).optional().describe('Email categories'),
  customArgs: z.record(z.unknown()).optional().describe('Custom arguments'),

  // Scheduling
  sendAt: z.union([z.date(), z.number()]).optional().describe('Schedule timestamp (Date or Unix timestamp)'),
  batchId: z.string().optional().describe('Batch ID for batch sending'),

  // List operations
  listName: z.string().optional().describe('List name'),
  listId: z.number().optional().describe('List ID'),
  recipients: z.array(z.object({
    email: z.string().email(),
    firstName: z.string().optional(),
    lastName: z.string().optional(),
    customFields: z.record(z.unknown()).optional(),
  })).optional().describe('Recipients for bulk operations'),

  // Template operations
  templateGeneration: z.enum(['legacy', 'dynamic']).default('dynamic').describe('Template generation'),

  // Pagination
  limit: z.number().min(1).max(1000).default(100).describe('Number of results to return'),
  offset: z.number().optional().describe('Pagination offset'),

  // Date range for stats
  startDate: z.string().optional().describe('Start date (YYYY-MM-DD)'),
  endDate: z.string().optional().describe('End date (YYYY-MM-DD)'),

  // Timeout
  timeout: z.number().min(1000).max(120000).default(30000).describe('Request timeout in ms'),
});

type SendGridParamsInput = z.input<typeof SendGridParamsSchema>;
type SendGridParams = z.output<typeof SendGridParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const SendGridResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  status: z.object({
    code: z.number(),
    reason: z.string().optional(),
  }),
  error: z.string().optional(),
  timing: z.number().describe('Response time in ms'),
  messageId: z.string().optional(),
  headers: z.record(z.string()).optional(),
  pagination: z.object({
    nextOffset: z.number().optional(),
    totalCount: z.number().optional(),
  }).optional(),
});

type SendGridResult = z.output<typeof SendGridResultSchema>;

// ============================================================================
// SENDGRID BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class SendGridBubble extends ServiceBubble<SendGridParams, SendGridResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'sendgrid' as const;
  static readonly type = 'service' as const;
  static readonly schema = SendGridParamsSchema;
  static readonly resultSchema = SendGridResultSchema;
  static readonly credentialType = 'sendgrid_api_key' as const;

  static readonly shortDescription = 'SendGrid API integration for email operations';
  static readonly longDescription = `
    SendGrid API service bubble for email operations.

    Features:
    - Send single and bulk emails
    - Template-based emails
    - List management
    - Email scheduling
    - Validation and analytics
    - Bounces and delivery status
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - apiKey: SendGrid API key (no default - must be provided)
    - baseUrl: SendGrid API base URL (defaults to https://api.sendgrid.com/v3)

    Federation Constitution Compliance:
    - No magic defaults (apiKey is required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: SendGridParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    SendGridBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('sendgrid', DEFAULT_RESILIENCE_CONFIG);
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - apiKey is required by schema
  }

  /**
   * Build HTTP headers for SendGrid API requests
   */
  private buildHeaders(): Record<string, string> {
    return {
      'Authorization': `Bearer ${this.params.apiKey}`,
      'Content-Type': 'application/json',
    };
  }

  /**
   * Build full URL for SendGrid API endpoint
   */
  private buildUrl(endpoint: string): string {
    return `${this.params.baseUrl}/${endpoint}`;
  }

  /**
   * Make HTTP request to SendGrid API
   */
  private async makeRequest(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<{ response: Response; data: any; timing: number; headers: Record<string, string> }> {
    const startTime = Date.now();
    const url = this.buildUrl(endpoint);

    const response = await fetch(url, {
      method,
      headers: this.buildHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });

    const timing = Date.now() - startTime;

    // Extract response headers
    const headers: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      headers[key] = value;
    });

    let data: any;
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    return { response, data, timing, headers };
  }

  /**
   * Send email operation
   */
  private async sendEmail(): Promise<SendGridResult> {
    if (!this.params.to || !this.params.from) {
      throw new Error('to and from are required for send_email operation');
    }

    const startTime = Date.now();

    try {
      const to = Array.isArray(this.params.to) ? this.params.to : [this.params.to];

      const personalizations = to.map(email => ({
        to: [{ email }],
        ...(this.params.dynamicData && { dynamicTemplateData: this.params.dynamicData }),
      }));

      const body: Record<string, unknown> = {
        personalizations,
        from: { email: this.params.from },
        subject: this.params.subject || '',
        ...(this.params.text && { content: [{ type: 'text/plain', value: this.params.text }] }),
        ...(this.params.html && { content: [{ type: 'text/html', value: this.params.html }] }),
        ...(this.params.attachments && { attachments: this.params.attachments }),
        ...(this.params.categories && { categories: this.params.categories }),
        ...(this.params.customArgs && { custom_args: this.params.customArgs }),
      };

      // Handle scheduling
      if (this.params.sendAt) {
        const sendAt = this.params.sendAt instanceof Date
          ? Math.floor(this.params.sendAt.getTime() / 1000)
          : this.params.sendAt;
        body.send_at = sendAt;
      }

      // Handle batch ID
      if (this.params.batchId) {
        body.batch_id = this.params.batchId;
      }

      const { response, data, timing, headers } = await this.resilience.execute(
        `sendgrid-send-email-${to.join(',')}`,
        () => this.makeRequest('POST', 'mail/send', body),
        { operation: 'send_email', to: this.params.to }
      );

      const messageId = headers['x-message-id'];

      return {
        success: response.ok || response.status === 202,
        operation: 'send_email',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || data?.message || 'Unknown error'),
        timing,
        messageId,
        headers,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'send_email',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Send bulk email operation
   */
  private async sendBulkEmail(): Promise<SendGridResult> {
    if (!this.params.recipients || this.params.recipients.length === 0) {
      throw new Error('recipients are required for send_bulk_email operation');
    }

    const startTime = Date.now();

    try {
      const personalizations = this.params.recipients.map(recipient => ({
        to: [{ email: recipient.email, name: `${recipient.firstName || ''} ${recipient.lastName || ''}`.trim() || undefined }],
        ...(recipient.customFields && { customArgs: recipient.customFields }),
      }));

      const body: Record<string, unknown> = {
        personalizations,
        from: { email: this.params.from },
        subject: this.params.subject || '',
        ...(this.params.text && { content: [{ type: 'text/plain', value: this.params.text }] }),
        ...(this.params.html && { content: [{ type: 'text/html', value: this.params.html }] }),
        ...(this.params.templateId && { template_id: this.params.templateId }),
      };

      const { response, data, timing, headers } = await this.resilience.execute(
        `sendgrid-send-bulk-email-${this.params.recipients.length}`,
        () => this.makeRequest('POST', 'mail/send', body),
        { operation: 'send_bulk_email', recipientCount: this.params.recipients.length }
      );

      return {
        success: response.ok || response.status === 202,
        operation: 'send_bulk_email',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
        messageId: headers['x-message-id'],
        headers,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'send_bulk_email',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Add recipient to list operation
   */
  private async addRecipientToList(): Promise<SendGridResult> {
    if (!this.params.listId || !this.params.recipients || this.params.recipients.length === 0) {
      throw new Error('listId and recipients are required for add_recipient_to_list operation');
    }

    const startTime = Date.now();

    try {
      const body = {
        list_id: this.params.listId,
        contacts: this.params.recipients.map(r => ({
          email: r.email,
          first_name: r.firstName,
          last_name: r.lastName,
          ...r.customFields,
        })),
      };

      const { response, data, timing } = await this.resilience.execute(
        `sendgrid-add-recipient-list-${this.params.listId}`,
        () => this.makeRequest('POST', 'marketing/contacts', body),
        { operation: 'add_recipient_to_list', listId: this.params.listId }
      );

      return {
        success: response.ok || response.status === 202,
        operation: 'add_recipient_to_list',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'add_recipient_to_list',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Create list operation
   */
  private async createList(): Promise<SendGridResult> {
    if (!this.params.listName) {
      throw new Error('listName is required for create_list operation');
    }

    const startTime = Date.now();

    try {
      const body = {
        name: this.params.listName,
      };

      const { response, data, timing } = await this.resilience.execute(
        `sendgrid-create-list-${this.params.listName}`,
        () => this.makeRequest('POST', 'marketing/lists', body),
        { operation: 'create_list', listName: this.params.listName }
      );

      return {
        success: response.ok || response.status === 201,
        operation: 'create_list',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'create_list',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get templates operation
   */
  private async getTemplates(): Promise<SendGridResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        page_size: String(this.params.limit),
        ...(this.params.offset !== undefined && { page_offset: String(this.params.offset) }),
        generations: this.params.templateGeneration,
      });

      const { response, data, timing } = await this.resilience.execute(
        'sendgrid-get-templates',
        () => this.makeRequest('GET', `templates?${params.toString()}`),
        { operation: 'get_templates' }
      );

      return {
        success: response.ok,
        operation: 'get_templates',
        data: data?.templates,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_templates',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Send with template operation
   */
  private async sendWithTemplate(): Promise<SendGridResult> {
    if (!this.params.to || !this.params.from || !this.params.templateId) {
      throw new Error('to, from, and templateId are required for send_with_template operation');
    }

    const startTime = Date.now();

    try {
      const to = Array.isArray(this.params.to) ? this.params.to : [this.params.to];

      const personalizations = to.map(email => ({
        to: [{ email }],
        ...(this.params.dynamicData && { dynamicTemplateData: this.params.dynamicData }),
      }));

      const body = {
        personalizations,
        from: { email: this.params.from },
        template_id: this.params.templateId,
      };

      const { response, data, timing, headers } = await this.resilience.execute(
        `sendgrid-send-template-${this.params.templateId}`,
        () => this.makeRequest('POST', 'mail/send', body),
        { operation: 'send_with_template', templateId: this.params.templateId }
      );

      return {
        success: response.ok || response.status === 202,
        operation: 'send_with_template',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
        messageId: headers['x-message-id'],
        headers,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'send_with_template',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Validate email operation
   */
  private async validateEmail(): Promise<SendGridResult> {
    if (!this.params.to || typeof this.params.to !== 'string') {
      throw new Error('to (single email) is required for validate_email operation');
    }

    const startTime = Date.now();

    try {
      const body = {
        email: this.params.to,
      };

      const { response, data, timing } = await this.resilience.execute(
        `sendgrid-validate-email-${this.params.to}`,
        () => this.makeRequest('POST', 'validations/email', body),
        { operation: 'validate_email', email: this.params.to }
      );

      return {
        success: response.ok || response.status === 202,
        operation: 'validate_email',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'validate_email',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Schedule email operation
   */
  private async scheduleEmail(): Promise<SendGridResult> {
    if (!this.params.to || !this.params.from || !this.params.sendAt) {
      throw new Error('to, from, and sendAt are required for schedule_email operation');
    }

    const startTime = Date.now();

    try {
      const sendAt = this.params.sendAt instanceof Date
        ? Math.floor(this.params.sendAt.getTime() / 1000)
        : this.params.sendAt;

      const to = Array.isArray(this.params.to) ? this.params.to : [this.params.to];

      const personalizations = to.map(email => ({
        to: [{ email }],
        sendAt,
      }));

      const body: Record<string, unknown> = {
        personalizations,
        from: { email: this.params.from },
        subject: this.params.subject || '',
        send_at: sendAt,
        ...(this.params.text && { content: [{ type: 'text/plain', value: this.params.text }] }),
        ...(this.params.html && { content: [{ type: 'text/html', value: this.params.html }] }),
      };

      const { response, data, timing } = await this.resilience.execute(
        `sendgrid-schedule-email-${sendAt}`,
        () => this.makeRequest('POST', 'mail/send', body),
        { operation: 'schedule_email', sendAt }
      );

      return {
        success: response.ok || response.status === 202,
        operation: 'schedule_email',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'schedule_email',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Cancel scheduled email operation
   */
  private async cancelScheduledEmail(): Promise<SendGridResult> {
    if (!this.params.batchId) {
      throw new Error('batchId is required for cancel_scheduled_email operation');
    }

    const startTime = Date.now();

    try {
      const { response, data, timing } = await this.resilience.execute(
        `sendgrid-cancel-scheduled-${this.params.batchId}`,
        () => this.makeRequest('POST', `user/scheduled_sends/${this.params.batchId}`, undefined),
        { operation: 'cancel_scheduled_email', batchId: this.params.batchId }
      );

      return {
        success: response.ok || response.status === 204,
        operation: 'cancel_scheduled_email',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'cancel_scheduled_email',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get stats operation
   */
  private async getStats(): Promise<SendGridResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        ...(this.params.startDate && { start_date: this.params.startDate }),
        ...(this.params.endDate && { end_date: this.params.endDate }),
        aggregated_by: 'day',
      });

      const { response, data, timing } = await this.resilience.execute(
        'sendgrid-get-stats',
        () => this.makeRequest('GET', `stats?${params.toString()}`),
        { operation: 'get_stats' }
      );

      return {
        success: response.ok,
        operation: 'get_stats',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_stats',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get bounces operation
   */
  private async getBounces(): Promise<SendGridResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        limit: String(this.params.limit),
        ...(this.params.offset !== undefined && { offset: String(this.params.offset) }),
      });

      const { response, data, timing } = await this.resilience.execute(
        'sendgrid-get-bounces',
        () => this.makeRequest('GET', `suppression/bounces?${params.toString()}`),
        { operation: 'get_bounces' }
      );

      return {
        success: response.ok,
        operation: 'get_bounces',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
        pagination: {
          nextOffset: data?._metadata?.next_offset,
          totalCount: data?._metadata?.total,
        },
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_bounces',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get delivery status operation
   */
  private async getDeliveryStatus(): Promise<SendGridResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        limit: String(this.params.limit),
        ...(this.params.offset !== undefined && { offset: String(this.params.offset) }),
      });

      const { response, data, timing } = await this.resilience.execute(
        'sendgrid-get-delivery-status',
        () => this.makeRequest('GET', `messages?${params.toString()}`),
        { operation: 'get_delivery_status' }
      );

      return {
        success: response.ok,
        operation: 'get_delivery_status',
        data: data?.messages,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data?.errors?.[0]?.message || 'Unknown error'),
        timing,
        pagination: {
          nextOffset: data?._metadata?.next,
          totalCount: data?._metadata?.total_count,
        },
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_delivery_status',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<SendGridResult> {
    switch (this.params.operation) {
      case 'send_email':
        return this.sendEmail();
      case 'send_bulk_email':
        return this.sendBulkEmail();
      case 'add_recipient_to_list':
        return this.addRecipientToList();
      case 'create_list':
        return this.createList();
      case 'get_templates':
        return this.getTemplates();
      case 'send_with_template':
        return this.sendWithTemplate();
      case 'validate_email':
        return this.validateEmail();
      case 'schedule_email':
        return this.scheduleEmail();
      case 'cancel_scheduled_email':
        return this.cancelScheduledEmail();
      case 'get_stats':
        return this.getStats();
      case 'get_bounces':
        return this.getBounces();
      case 'get_delivery_status':
        return this.getDeliveryStatus();
      default:
        return {
          success: false,
          operation: this.params.operation,
          status: { code: 400, reason: 'Invalid operation' },
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default SendGridBubble;
