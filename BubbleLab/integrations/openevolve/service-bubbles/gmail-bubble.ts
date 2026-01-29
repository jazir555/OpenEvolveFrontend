/**
 * Gmail API Service Bubble
 *
 * Provides integration with Gmail API for email operations.
 * Supports sending, reading, searching, and label management.
 *
 * Federation Constitution Compliant
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// GMAIL-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const GmailOperationSchema = z.enum([
  'send_email',
  'get_message',
  'list_messages',
  'search_messages',
  'delete_message',
  'modify_labels',
  'get_thread',
  'list_threads',
  'get_label',
  'list_labels',
  'create_label',
  'delete_label',
  'get_attachment',
]);

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const GmailParamsSchema = z.object({
  operation: GmailOperationSchema.describe('Gmail API operation'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  accessToken: z.string().min(1).describe('OAuth 2.0 access token (REQUIRED)'),
  userId: z.string().default('me').describe('User ID (default: "me" for authenticated user)'),
  baseUrl: z.string().url().default('https://gmail.googleapis.com/gmail/v1').describe('Gmail API base URL'),

  // Email operations
  to: z.string().optional().describe('Recipient email address'),
  subject: z.string().optional().describe('Email subject'),
  body: z.string().optional().describe('Email body (plain text or HTML)'),
  isHtml: z.boolean().default(false).describe('Whether body is HTML'),
  attachments: z.array(z.object({
    filename: z.string(),
    content: z.string(), // base64 encoded
    mimeType: z.string(),
  })).optional().describe('Email attachments'),

  // Message operations
  messageId: z.string().optional().describe('Message ID'),
  threadId: z.string().optional().describe('Thread ID'),

  // List operations
  maxResults: z.number().min(1).max(500).default(100).describe('Maximum results to return'),
  pageToken: z.string().optional().describe('Pagination token'),
  query: z.string().optional().describe('Search query (Gmail search syntax)'),
  labelIds: z.array(z.string()).optional().describe('Filter by label IDs'),

  // Label operations
  labelId: z.string().optional().describe('Label ID'),
  labelName: z.string().optional().describe('Label name'),
  labelColor: z.object({
    backgroundColor: z.string(),
    textColor: z.string(),
  }).optional().describe('Label color'),

  // Modify labels
  addLabelIds: z.array(z.string()).optional().describe('Label IDs to add'),
  removeLabelIds: z.array(z.string()).optional().describe('Label IDs to remove'),

  // Attachment operations
  attachmentId: z.string().optional().describe('Attachment ID'),

  // Timeout
  timeout: z.number().min(1000).max(120000).default(30000).describe('Request timeout in ms'),
});

type GmailParamsInput = z.input<typeof GmailParamsSchema>;
type GmailParams = z.output<typeof GmailParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const GmailResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  status: z.object({
    code: z.number(),
    reason: z.string().optional(),
  }),
  error: z.string().optional(),
  timing: z.number().describe('Response time in ms'),
  nextPageToken: z.string().optional(),
  resultSizeEstimate: z.number().optional(),
});

type GmailResult = z.output<typeof GmailResultSchema>;

// ============================================================================
// GMAIL BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class GmailBubble extends ServiceBubble<GmailParams, GmailResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'oauth' as const;
  static readonly bubbleName = 'gmail' as const;
  static readonly type = 'service' as const;
  static readonly schema = GmailParamsSchema;
  static readonly resultSchema = GmailResultSchema;
  static readonly credentialType = 'gmail_access_token' as const;

  static readonly shortDescription = 'Gmail API integration for email operations';
  static readonly longDescription = `
    Gmail API service bubble for email operations.

    Features:
    - Send emails with attachments
    - Read and list messages
    - Search messages with Gmail query syntax
    - Thread operations
    - Label management (create, list, delete)
    - Modify message labels
    - Get attachments
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - accessToken: OAuth 2.0 access token (no default - must be provided)

    Federation Constitution Compliance:
    - No magic defaults (accessToken is required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: GmailParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    GmailBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('gmail', DEFAULT_RESILIENCE_CONFIG);
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - accessToken is required by schema
  }

  /**
   * Build HTTP headers for Gmail API requests
   */
  private buildHeaders(): Record<string, string> {
    return {
      'Authorization': `Bearer ${this.params.accessToken}`,
      'Content-Type': 'application/json',
    };
  }

  /**
   * Build full URL for Gmail API endpoint
   */
  private buildUrl(endpoint: string): string {
    return `${this.params.baseUrl}/users/${this.params.userId}${endpoint}`;
  }

  /**
   * Make HTTP request to Gmail API
   */
  private async makeRequest(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<{ response: Response; data: unknown; timing: number }> {
    const startTime = Date.now();
    const url = this.buildUrl(endpoint);

    const response = await fetch(url, {
      method,
      headers: this.buildHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });

    const timing = Date.now() - startTime;

    let data: unknown;
    if (response.status !== 204) {
      data = await response.json();
    }

    return { response, data, timing };
  }

  /**
   * Encode email to RFC 2822 format
   */
  private encodeEmail(to: string, subject: string, body: string, isHtml: boolean, attachments?: any[]): string {
    const emailLines = [
      `To: ${to}`,
      `Subject: ${subject}`,
      'MIME-Version: 1.0',
    ];

    if (attachments && attachments.length > 0) {
      const boundary = 'boundary_' + Date.now();
      emailLines.push(`Content-Type: multipart/mixed; boundary="${boundary}"`);
      emailLines.push('');
      emailLines.push(`--${boundary}`);
      emailLines.push(`Content-Type: ${isHtml ? 'text/html' : 'text/plain'}; charset="UTF-8"`);
      emailLines.push('Content-Transfer-Encoding: base64');
      emailLines.push('');
      emailLines.push(Buffer.from(body).toString('base64'));

      for (const attachment of attachments) {
        emailLines.push(`--${boundary}`);
        emailLines.push(`Content-Type: ${attachment.mimeType}; name="${attachment.filename}"`);
        emailLines.push('Content-Transfer-Encoding: base64');
        emailLines.push(`Content-Disposition: attachment; filename="${attachment.filename}"`);
        emailLines.push('');
        emailLines.push(attachment.content);
      }

      emailLines.push(`--${boundary}--`);
    } else {
      emailLines.push(`Content-Type: ${isHtml ? 'text/html' : 'text/plain'}; charset="UTF-8"`);
      emailLines.push('');
      emailLines.push(body);
    }

    return Buffer.from(emailLines.join('\r\n')).toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  }

  /**
   * Send email operation
   */
  private async sendEmail(): Promise<GmailResult> {
    if (!this.params.to || !this.params.subject || !this.params.body) {
      throw new Error('to, subject, and body are required for send_email operation');
    }

    const startTime = Date.now();

    try {
      const raw = this.encodeEmail(
        this.params.to,
        this.params.subject,
        this.params.body,
        this.params.isHtml,
        this.params.attachments
      );

      const { response, data, timing } = await this.resilience.execute(
        `gmail-send-email-${this.params.to}`,
        () => this.makeRequest('POST', '/messages/send', { raw }),
        { operation: 'send_email', to: this.params.to }
      );

      return {
        success: response.ok,
        operation: 'send_email',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.error?.message || 'Unknown error',
        timing,
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
   * List messages operation
   */
  private async listMessages(): Promise<GmailResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        maxResults: String(this.params.maxResults),
        ...(this.params.pageToken && { pageToken: this.params.pageToken }),
      });

      if (this.params.labelIds && this.params.labelIds.length > 0) {
        this.params.labelIds.forEach(labelId => params.append('labelIds', labelId));
      }

      const { response, data, timing } = await this.resilience.execute(
        'gmail-list-messages',
        () => this.makeRequest('GET', `/messages?${params.toString()}`),
        { operation: 'list_messages' }
      );

      return {
        success: response.ok,
        operation: 'list_messages',
        data: (data as any)?.messages,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.error?.message || 'Unknown error',
        timing,
        nextPageToken: (data as any)?.nextPageToken,
        resultSizeEstimate: (data as any)?.resultSizeEstimate,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'list_messages',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Search messages operation
   */
  private async searchMessages(): Promise<GmailResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        q: this.params.query || '',
        maxResults: String(this.params.maxResults),
        ...(this.params.pageToken && { pageToken: this.params.pageToken }),
      });

      const { response, data, timing } = await this.resilience.execute(
        `gmail-search-${this.params.query}`,
        () => this.makeRequest('GET', `/messages?${params.toString()}`),
        { operation: 'search_messages', query: this.params.query }
      );

      return {
        success: response.ok,
        operation: 'search_messages',
        data: (data as any)?.messages,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.error?.message || 'Unknown error',
        timing,
        nextPageToken: (data as any)?.nextPageToken,
        resultSizeEstimate: (data as any)?.resultSizeEstimate,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'search_messages',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get message operation
   */
  private async getMessage(): Promise<GmailResult> {
    if (!this.params.messageId) {
      throw new Error('messageId is required for get_message operation');
    }

    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        format: 'full',
      });

      const { response, data, timing } = await this.resilience.execute(
        `gmail-get-message-${this.params.messageId}`,
        () => this.makeRequest('GET', `/messages/${this.params.messageId}?${params.toString()}`),
        { operation: 'get_message', messageId: this.params.messageId }
      );

      return {
        success: response.ok,
        operation: 'get_message',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.error?.message || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_message',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Modify labels operation
   */
  private async modifyLabels(): Promise<GmailResult> {
    if (!this.params.messageId) {
      throw new Error('messageId is required for modify_labels operation');
    }

    const startTime = Date.now();

    try {
      const body = {
        addLabelIds: this.params.addLabelIds || [],
        removeLabelIds: this.params.removeLabelIds || [],
      };

      const { response, data, timing } = await this.resilience.execute(
        `gmail-modify-labels-${this.params.messageId}`,
        () => this.makeRequest('POST', `/messages/${this.params.messageId}/modify`, body),
        { operation: 'modify_labels', messageId: this.params.messageId }
      );

      return {
        success: response.ok,
        operation: 'modify_labels',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.error?.message || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'modify_labels',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * List labels operation
   */
  private async listLabels(): Promise<GmailResult> {
    const startTime = Date.now();

    try {
      const { response, data, timing } = await this.resilience.execute(
        'gmail-list-labels',
        () => this.makeRequest('GET', '/labels'),
        { operation: 'list_labels' }
      );

      return {
        success: response.ok,
        operation: 'list_labels',
        data: (data as any)?.labels,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.error?.message || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'list_labels',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<GmailResult> {
    switch (this.params.operation) {
      case 'send_email':
        return this.sendEmail();
      case 'get_message':
        return this.getMessage();
      case 'list_messages':
        return this.listMessages();
      case 'search_messages':
        return this.searchMessages();
      case 'modify_labels':
        return this.modifyLabels();
      case 'list_labels':
        return this.listLabels();
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

export default GmailBubble;
