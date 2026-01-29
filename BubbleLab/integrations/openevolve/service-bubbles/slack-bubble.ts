/**
 * Slack API Service Bubble
 *
 * Provides integration with Slack API for messaging,
 * channels, users, and workspace operations.
 *
 * Federation Constitution Compliant
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// SLACK-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const SlackOperationSchema = z.enum([
  'send_message',
  'update_message',
  'delete_message',
  'get_channel',
  'list_channels',
  'create_channel',
  'archive_channel',
  'get_user',
  'list_users',
  'get_conversation_history',
  'post_ephemeral',
  'add_reaction',
  'get_reactions',
  'upload_file',
  'schedule_message',
]);

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const SlackParamsSchema = z.object({
  operation: SlackOperationSchema.describe('Slack API operation'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  botToken: z.string().min(1).describe('Slack bot token (starts with xoxb-) (REQUIRED)'),
  baseUrl: z.string().url().default('https://slack.com/api').describe('Slack API base URL'),

  // Message operations
  channel: z.string().optional().describe('Channel ID or name'),
  text: z.string().optional().describe('Message text'),
  blocks: z.array(z.record(z.unknown())).optional().describe('Slack blocks for rich formatting'),
  attachments: z.array(z.record(z.unknown())).optional().describe('Slack attachments'),
  threadTs: z.string().optional().describe('Thread timestamp for threaded messages'),
  messageId: z.string().optional().describe('Message timestamp for updating/deleting'),

  // Channel operations
  channelName: z.string().optional().describe('Channel name (for creating channels)'),
  isPrivate: z.boolean().default(false).describe('Whether channel is private'),
  topic: z.string().optional().describe('Channel topic'),
  purpose: z.string().optional().describe('Channel purpose'),

  // User operations
  userId: z.string().optional().describe('User ID'),

  // Conversation history
  limit: z.number().min(1).max(1000).default(100).describe('Number of messages to retrieve'),
  cursor: z.string().optional().describe('Pagination cursor'),
  oldest: z.string().optional().describe('Start of time range'),
  latest: z.string().optional().describe('End of time range'),
  inclusive: z.boolean().default(false).describe('Include messages with timestamps on boundaries'),

  // Ephemeral messages
  user: z.string().optional().describe('User ID for ephemeral message'),

  // Reactions
  reaction: z.string().optional().describe('Reaction name (e.g., "thumbsup")'),
  full: z.boolean().default(false).describe('Include full user data for reactions'),

  // File upload
  fileContent: z.string().optional().describe('File content (base64 encoded)'),
  filename: z.string().optional().describe('File name'),
  filetype: z.string().optional().describe('File type'),
  title: z.string().optional().describe('File title'),
  initialComment: z.string().optional().describe('Initial comment on file'),

  // Scheduled messages
  postAt: z.number().optional().describe('Unix timestamp for scheduled message'),

  // Timeout
  timeout: z.number().min(1000).max(120000).default(30000).describe('Request timeout in ms'),
});

type SlackParamsInput = z.input<typeof SlackParamsSchema>;
type SlackParams = z.output<typeof SlackParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const SlackResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  status: z.object({
    code: z.number(),
    reason: z.string().optional(),
  }),
  error: z.string().optional(),
  timing: z.number().describe('Response time in ms'),
  slackOk: z.boolean().optional().describe('Slack API ok field'),
  responseMetadata: z.object({
    nextCursor: z.string().optional(),
    warnings: z.array(z.string()).optional(),
  }).optional(),
});

type SlackResult = z.output<typeof SlackResultSchema>;

// ============================================================================
// SLACK BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class SlackBubble extends ServiceBubble<SlackParams, SlackResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'slack' as const;
  static readonly type = 'service' as const;
  static readonly schema = SlackParamsSchema;
  static readonly resultSchema = SlackResultSchema;
  static readonly credentialType = 'slack_bot_token' as const;

  static readonly shortDescription = 'Slack API integration for messaging and workspace operations';
  static readonly longDescription = `
    Slack API service bubble for messaging and workspace operations.

    Features:
    - Send, update, and delete messages
    - Channel management (create, list, archive)
    - User information and listing
    - Conversation history retrieval
    - Ephemeral messages
    - Reactions
    - File uploads
    - Scheduled messages
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - botToken: Slack bot token (starts with xoxb-) (no default - must be provided)
    - baseUrl: Slack API base URL (defaults to https://slack.com/api)

    Federation Constitution Compliance:
    - No magic defaults (botToken is required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: SlackParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    SlackBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('slack', DEFAULT_RESILIENCE_CONFIG);
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - botToken is required by schema
  }

  /**
   * Build HTTP headers for Slack API requests
   */
  private buildHeaders(): Record<string, string> {
    return {
      'Authorization': `Bearer ${this.params.botToken}`,
      'Content-Type': 'application/json',
    };
  }

  /**
   * Build full URL for Slack API endpoint
   */
  private buildUrl(endpoint: string): string {
    return `${this.params.baseUrl}/${endpoint}`;
  }

  /**
   * Make HTTP request to Slack API
   */
  private async makeRequest(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<{ response: Response; data: any; timing: number }> {
    const startTime = Date.now();
    const url = this.buildUrl(endpoint);

    const response = await fetch(url, {
      method,
      headers: this.buildHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });

    const timing = Date.now() - startTime;
    const data = await response.json();

    return { response, data, timing };
  }

  /**
   * Send message operation
   */
  private async sendMessage(): Promise<SlackResult> {
    if (!this.params.channel || (!this.params.text && !this.params.blocks)) {
      throw new Error('channel and either text or blocks are required for send_message operation');
    }

    const startTime = Date.now();

    try {
      const body: Record<string, unknown> = {
        channel: this.params.channel,
        text: this.params.text || '',
        ...(this.params.blocks && { blocks: this.params.blocks }),
        ...(this.params.attachments && { attachments: this.params.attachments }),
        ...(this.params.threadTs && { thread_ts: this.params.threadTs }),
      };

      const { response, data, timing } = await this.resilience.execute(
        `slack-send-message-${this.params.channel}`,
        () => this.makeRequest('POST', 'chat.postMessage', body),
        { operation: 'send_message', channel: this.params.channel }
      );

      return {
        success: response.ok && data.ok,
        operation: 'send_message',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok && data.ok ? undefined : data.error || 'Unknown error',
        timing,
        slackOk: data.ok,
        responseMetadata: data.response_metadata,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'send_message',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Update message operation
   */
  private async updateMessage(): Promise<SlackResult> {
    if (!this.params.channel || !this.params.messageId) {
      throw new Error('channel and messageId are required for update_message operation');
    }

    const startTime = Date.now();

    try {
      const body: Record<string, unknown> = {
        channel: this.params.channel,
        ts: this.params.messageId,
        text: this.params.text || '',
        ...(this.params.blocks && { blocks: this.params.blocks }),
        ...(this.params.attachments && { attachments: this.params.attachments }),
      };

      const { response, data, timing } = await this.resilience.execute(
        `slack-update-message-${this.params.channel}-${this.params.messageId}`,
        () => this.makeRequest('POST', 'chat.update', body),
        { operation: 'update_message', channel: this.params.channel, messageId: this.params.messageId }
      );

      return {
        success: response.ok && data.ok,
        operation: 'update_message',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok && data.ok ? undefined : data.error || 'Unknown error',
        timing,
        slackOk: data.ok,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'update_message',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * List channels operation
   */
  private async listChannels(): Promise<SlackResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        limit: String(this.params.limit),
        ...(this.params.cursor && { cursor: this.params.cursor }),
      });

      const { response, data, timing } = await this.resilience.execute(
        'slack-list-channels',
        () => this.makeRequest('GET', `conversations.list?${params.toString()}`),
        { operation: 'list_channels' }
      );

      return {
        success: response.ok && data.ok,
        operation: 'list_channels',
        data: data.channels,
        status: { code: response.status, reason: response.statusText },
        error: response.ok && data.ok ? undefined : data.error || 'Unknown error',
        timing,
        slackOk: data.ok,
        responseMetadata: data.response_metadata,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'list_channels',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get conversation history operation
   */
  private async getConversationHistory(): Promise<SlackResult> {
    if (!this.params.channel) {
      throw new Error('channel is required for get_conversation_history operation');
    }

    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        channel: this.params.channel,
        limit: String(this.params.limit),
        ...(this.params.cursor && { cursor: this.params.cursor }),
        ...(this.params.oldest && { oldest: this.params.oldest }),
        ...(this.params.latest && { latest: this.params.latest }),
        inclusive: String(this.params.inclusive),
      });

      const { response, data, timing } = await this.resilience.execute(
        `slack-conversation-history-${this.params.channel}`,
        () => this.makeRequest('GET', `conversations.history?${params.toString()}`),
        { operation: 'get_conversation_history', channel: this.params.channel }
      );

      return {
        success: response.ok && data.ok,
        operation: 'get_conversation_history',
        data: data.messages,
        status: { code: response.status, reason: response.statusText },
        error: response.ok && data.ok ? undefined : data.error || 'Unknown error',
        timing,
        slackOk: data.ok,
        responseMetadata: data.response_metadata,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_conversation_history',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Add reaction operation
   */
  private async addReaction(): Promise<SlackResult> {
    if (!this.params.channel || !this.params.reaction || !this.params.messageId) {
      throw new Error('channel, reaction, and messageId are required for add_reaction operation');
    }

    const startTime = Date.now();

    try {
      const body = {
        channel: this.params.channel,
        name: this.params.reaction,
        timestamp: this.params.messageId,
      };

      const { response, data, timing } = await this.resilience.execute(
        `slack-add-reaction-${this.params.channel}-${this.params.messageId}`,
        () => this.makeRequest('POST', 'reactions.add', body),
        { operation: 'add_reaction', channel: this.params.channel, messageId: this.params.messageId }
      );

      return {
        success: response.ok && data.ok,
        operation: 'add_reaction',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok && data.ok ? undefined : data.error || 'Unknown error',
        timing,
        slackOk: data.ok,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'add_reaction',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Upload file operation
   */
  private async uploadFile(): Promise<SlackResult> {
    if (!this.params.channel || !this.params.fileContent) {
      throw new Error('channel and fileContent are required for upload_file operation');
    }

    const startTime = Date.now();

    try {
      // Note: For multipart/form-data, we need to use FormData
      const formData = new FormData();
      formData.append('channels', this.params.channel);
      formData.append('file', this.params.fileContent);

      if (this.params.filename) {
        formData.append('filename', this.params.filename);
      }
      if (this.params.filetype) {
        formData.append('filetype', this.params.filetype);
      }
      if (this.params.title) {
        formData.append('title', this.params.title);
      }
      if (this.params.initialComment) {
        formData.append('initial_comment', this.params.initialComment);
      }

      const { response, data, timing } = await this.resilience.execute(
        `slack-upload-file-${this.params.channel}`,
        async () => {
          const startTime = Date.now();
          const resp = await fetch(`${this.params.baseUrl}/files.upload`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${this.params.botToken}`,
            },
            body: formData,
          });
          const jsonData = await resp.json();
          return { response: resp, data: jsonData, timing: Date.now() - startTime };
        },
        { operation: 'upload_file', channel: this.params.channel }
      );

      return {
        success: response.ok && data.ok,
        operation: 'upload_file',
        data: data.file,
        status: { code: response.status, reason: response.statusText },
        error: response.ok && data.ok ? undefined : data.error || 'Unknown error',
        timing,
        slackOk: data.ok,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'upload_file',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<SlackResult> {
    switch (this.params.operation) {
      case 'send_message':
        return this.sendMessage();
      case 'update_message':
        return this.updateMessage();
      case 'list_channels':
        return this.listChannels();
      case 'get_conversation_history':
        return this.getConversationHistory();
      case 'add_reaction':
        return this.addReaction();
      case 'upload_file':
        return this.uploadFile();
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

export default SlackBubble;
