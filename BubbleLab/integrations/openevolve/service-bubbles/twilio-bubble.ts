/**
 * Twilio API Service Bubble
 *
 * Provides integration with Twilio API for SMS, voice, and phone operations.
 * Supports messaging, calls, phone numbers, and account management.
 *
 * Federation Constitution Compliant
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// TWILIO-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const TwilioOperationSchema = z.enum([
  'send_sms',
  'send_bulk_sms',
  'make_call',
  'get_call_status',
  'record_call',
  'get_call_recording',
  'get_phone_number',
  'buy_phone_number',
  'release_phone_number',
  'get_messages',
  'get_account_info',
  'get_usage',
]);

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const TwilioParamsSchema = z.object({
  operation: TwilioOperationSchema.describe('Twilio API operation'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  accountSid: z.string().min(1).describe('Twilio Account SID (REQUIRED)'),
  apiKey: z.string().min(1).describe('Twilio API key or auth token (REQUIRED)'),
  baseUrl: z.string().url().default('https://api.twilio.com').describe('Twilio API base URL'),

  // SMS operations
  to: z.union([z.string(), z.array(z.string())]).optional().describe('Recipient phone number(s)'),
  from: z.string().optional().describe('Sender phone number'),
  message: z.string().optional().describe('SMS message body'),
  messagingServiceSid: z.string().optional().describe('Messaging service SID for bulk sending'),

  // Call operations
  url: z.string().url().optional().describe('TwiML URL for call handling'),
  twiml: z.string().optional().describe('TwiML instructions for call'),
  statusCallback: z.string().url().optional().describe('Status callback URL'),
  statusCallbackEvent: z.array(z.string()).optional().describe('Status callback events'),
  record: z.boolean().default(false).describe('Record the call'),
  timeout: z.number().int().positive().default(30).describe('Call timeout in seconds'),
  method: z.enum(['GET', 'POST']).default('POST').describe('HTTP method for URL'),

  // Recording operations
  callSid: z.string().optional().describe('Call SID'),
  recordingSid: z.string().optional().describe('Recording SID'),
  recordingStatus: z.enum(['in-progress', 'completed', 'absent').optional().describe('Recording status filter'),

  // Phone number operations
  phoneNumber: z.string().optional().describe('Phone number in E.164 format'),
  countryCode: z.string().length(2).optional().describe('ISO country code (e.g., US)'),
  areaCode: z.string().optional().describe('Area code for phone number search'),
  contains: z.string().optional().describe('Pattern phone number must contain'),
  voiceUrl: z.string().url().optional().describe('Voice URL for phone number'),

  // Pagination
  limit: z.number().min(1).max(1000).default(100).describe('Number of results to return'),
  pageSize: z.number().min(1).max(1000).default(50).describe('Page size for pagination'),
  pageToken: z.string().optional().describe('Pagination token'),

  // Date range filtering
  startDate: z.string().optional().describe('Start date (YYYY-MM-DD)'),
  endDate: z.string().optional().describe('End date (YYYY-MM-DD)'),

  // Usage operations
  category: z.string().optional().describe('Usage category'),
  usageCategory: z.enum([
    'calls',
    'sms',
    'mms',
    'phone_numbers',
    'short_codes',
    'callerid_lookups',
    'recordings',
    'transcriptions',
  ]).optional().describe('Usage category filter'),

  // Request timeout
  timeout: z.number().min(1000).max(120000).default(30000).describe('Request timeout in ms'),
});

type TwilioParamsInput = z.input<typeof TwilioParamsSchema>;
type TwilioParams = z.output<typeof TwilioParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const TwilioResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  status: z.object({
    code: z.number(),
    reason: z.string().optional(),
  }),
  error: z.string().optional(),
  timing: z.number().describe('Response time in ms'),
  sid: z.string().optional(),
  accountSid: z.string().optional(),
  pagination: z.object({
    nextPageUrl: z.string().optional(),
    nextPageToken: z.string().optional(),
    totalCount: z.number().optional(),
  }).optional(),
});

type TwilioResult = z.output<typeof TwilioResultSchema>;

// ============================================================================
// TWILIO BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class TwilioBubble extends ServiceBubble<TwilioParams, TwilioResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'twilio' as const;
  static readonly type = 'service' as const;
  static readonly schema = TwilioParamsSchema;
  static readonly resultSchema = TwilioResultSchema;
  static readonly credentialType = 'twilio_api_key' as const;

  static readonly shortDescription = 'Twilio API integration for SMS and voice operations';
  static readonly longDescription = `
    Twilio API service bubble for SMS, voice, and phone operations.

    Features:
    - Send single and bulk SMS
    - Make and manage voice calls
    - Call recording
    - Phone number management
    - Message and call history
    - Usage tracking and account info
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - accountSid: Twilio Account SID (no default - must be provided)
    - apiKey: Twilio API key or auth token (no default - must be provided)
    - baseUrl: Twilio API base URL (defaults to https://api.twilio.com)

    Federation Constitution Compliance:
    - No magic defaults (accountSid and apiKey are required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: TwilioParamsInput, context?: BubbleContext) {
    super(params, context);

    TwilioBubble.validateConfig();
    this.resilience = new ResilienceWrapper('twilio', DEFAULT_RESILIENCE_CONFIG);
  }

  private static validateConfig(): void {
    // Validation handled by schema
  }

  /**
   * Build HTTP headers with basic auth
   */
  private buildHeaders(): Record<string, string> {
    const credentials = Buffer.from(`${this.params.accountSid}:${this.params.apiKey}`).toString('base64');
    return {
      'Authorization': `Basic ${credentials}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    };
  }

  /**
   * Build full URL for Twilio API endpoint
   */
  private buildUrl(accountResource: string, subResource?: string, params?: string): string {
    let url = `${this.params.baseUrl}/2010-04-01/Accounts/${this.params.accountSid}/${accountResource}`;
    if (subResource) {
      url += `/${subResource}`;
    }
    if (params) {
      url += `?${params}`;
    }
    return url;
  }

  /**
   * Make HTTP request to Twilio API
   */
  private async makeRequest(
    method: string,
    url: string,
    body?: URLSearchParams
  ): Promise<{ response: Response; data: any; timing: number }> {
    const startTime = Date.now();

    const response = await fetch(url, {
      method,
      headers: {
        'Authorization': this.buildHeaders()['Authorization'],
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: body?.toString(),
    });

    const timing = Date.now() - startTime;

    let data: any;
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/xml')) {
      // Parse XML response (simplified - in production use proper XML parser)
      const text = await response.text();
      data = this.parseTwilioXml(text);
    } else if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    return { response, data, timing };
  }

  /**
   * Simple XML parser for Twilio responses (production should use proper XML library)
   */
  private parseTwilioXml(xml: string): any {
    // Very basic XML parsing - in production use xml2js or similar
    const result: any = {};

    const sidMatch = xml.match(/<Sid>(.+?)<\/Sid>/);
    if (sidMatch) result.sid = sidMatch[1];

    const statusMatch = xml.match(/<Status>(.+?)<\/Status>/);
    if (statusMatch) result.status = statusMatch[1];

    const accountSidMatch = xml.match(/<AccountSid>(.+?)<\/AccountSid>/);
    if (accountSidMatch) result.account_sid = accountSidMatch[1];

    const errorMessageMatch = xml.match(/<Message>(.+?)<\/Message>/);
    if (errorMessageMatch) result.error_message = errorMessageMatch[1];

    return result;
  }

  /**
   * Send SMS operation
   */
  private async sendSMS(): Promise<TwilioResult> {
    if (!this.params.to || !this.params.from || !this.params.message) {
      throw new Error('to, from, and message are required for send_sms operation');
    }

    const startTime = Date.now();

    try {
      const to = Array.isArray(this.params.to) ? this.params.to : [this.params.to];
      const results = [];

      for (const recipient of to) {
        const body = new URLSearchParams({
          To: recipient,
          From: this.params.from,
          Body: this.params.message,
        });

        const url = this.buildUrl('SMS/Messages');
        const { response, data, timing } = await this.resilience.execute(
          `twilio-send-sms-${recipient}`,
          () => this.makeRequest('POST', url, body),
          { operation: 'send_sms', to: recipient }
        );

        results.push({
          success: response.ok,
          data,
          status: { code: response.status, reason: response.statusText },
          sid: data?.sid,
        });
      }

      return {
        success: results.every(r => r.success),
        operation: 'send_sms',
        data: results,
        status: { code: 200, reason: 'OK' },
        timing: Date.now() - startTime,
        sid: results[0]?.sid,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'send_sms',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Send bulk SMS operation
   */
  private async sendBulkSMS(): Promise<TwilioResult> {
    if (!this.params.to || !this.params.message) {
      throw new Error('to and message are required for send_bulk_sms operation');
    }

    const startTime = Date.now();

    try {
      const to = Array.isArray(this.params.to) ? this.params.to : [this.params.to];

      if (this.params.messagingServiceSid) {
        // Use messaging service for bulk sending
        const body = new URLSearchParams({
          To: to.join(','),
          MessagingServiceSid: this.params.messagingServiceSid,
          Body: this.params.message,
        });

        const url = this.buildUrl('SMS/Messages');
        const { response, data, timing } = await this.resilience.execute(
          `twilio-send-bulk-sms-service`,
          () => this.makeRequest('POST', url, body),
          { operation: 'send_bulk_sms', recipientCount: to.length }
        );

        return {
          success: response.ok,
          operation: 'send_bulk_sms',
          data,
          status: { code: response.status, reason: response.statusText },
          error: response.ok ? undefined : data?.error_message,
          timing,
          sid: data?.sid,
        };
      } else if (this.params.from) {
        // Fallback to individual sends
        return this.sendSMS();
      } else {
        throw new Error('Either from or messagingServiceSid is required for bulk SMS');
      }
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'send_bulk_sms',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Make call operation
   */
  private async makeCall(): Promise<TwilioResult> {
    if (!this.params.to || !this.params.from || (!this.params.url && !this.params.twiml)) {
      throw new Error('to, from, and either url or twiml are required for make_call operation');
    }

    const startTime = Date.now();

    try {
      const body = new URLSearchParams({
        To: this.params.to,
        From: this.params.from,
        Url: this.params.url || '',
        Twiml: this.params.twiml || '',
        Method: this.params.method,
        Timeout: String(this.params.timeout),
      });

      if (this.params.statusCallback) {
        body.append('StatusCallback', this.params.statusCallback);
      }

      if (this.params.statusCallbackEvent && this.params.statusCallbackEvent.length > 0) {
        body.append('StatusCallbackEvent', this.params.statusCallbackEvent.join(' '));
      }

      if (this.params.record) {
        body.append('Record', 'true');
      }

      const url = this.buildUrl('Calls');
      const { response, data, timing } = await this.resilience.execute(
        `twilio-make-call-${this.params.to}`,
        () => this.makeRequest('POST', url, body),
        { operation: 'make_call', to: this.params.to }
      );

      return {
        success: response.ok,
        operation: 'make_call',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
        sid: data?.sid,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'make_call',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get call status operation
   */
  private async getCallStatus(): Promise<TwilioResult> {
    if (!this.params.callSid) {
      throw new Error('callSid is required for get_call_status operation');
    }

    const startTime = Date.now();

    try {
      const url = this.buildUrl('Calls', this.params.callSid);
      const { response, data, timing } = await this.resilience.execute(
        `twilio-get-call-${this.params.callSid}`,
        () => this.makeRequest('GET', url),
        { operation: 'get_call_status', callSid: this.params.callSid }
      );

      return {
        success: response.ok,
        operation: 'get_call_status',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
        sid: data?.sid,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_call_status',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Record call operation
   */
  private async recordCall(): Promise<TwilioResult> {
    if (!this.params.callSid) {
      throw new Error('callSid is required for record_call operation');
    }

    const startTime = Date.now();

    try {
      const body = new URLSearchParams({
        Record: 'true',
        RecordingStatus: this.params.recordingStatus || 'in-progress',
      });

      const url = this.buildUrl('Calls', this.params.callSid);
      const { response, data, timing } = await this.resilience.execute(
        `twilio-record-call-${this.params.callSid}`,
        () => this.makeRequest('POST', url, body),
        { operation: 'record_call', callSid: this.params.callSid }
      );

      return {
        success: response.ok,
        operation: 'record_call',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'record_call',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get call recording operation
   */
  private async getCallRecording(): Promise<TwilioResult> {
    if (!this.params.recordingSid && !this.params.callSid) {
      throw new Error('recordingSid or callSid is required for get_call_recording operation');
    }

    const startTime = Date.now();

    try {
      let url: string;

      if (this.params.recordingSid) {
        url = this.buildUrl('Recordings', this.params.recordingSid);
      } else {
        url = this.buildUrl('Calls', this.params.callSid, 'Recordings');
      }

      const { response, data, timing } = await this.resilience.execute(
        `twilio-get-recording-${this.params.recordingSid || this.params.callSid}`,
        () => this.makeRequest('GET', url),
        { operation: 'get_call_recording' }
      );

      return {
        success: response.ok,
        operation: 'get_call_recording',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_call_recording',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get phone number operation
   */
  private async getPhoneNumber(): Promise<TwilioResult> {
    if (!this.params.phoneNumber) {
      throw new Error('phoneNumber is required for get_phone_number operation');
    }

    const startTime = Date.now();

    try {
      const url = this.buildUrl('IncomingPhoneNumbers', this.params.phoneNumber);
      const { response, data, timing } = await this.resilience.execute(
        `twilio-get-phone-${this.params.phoneNumber}`,
        () => this.makeRequest('GET', url),
        { operation: 'get_phone_number', phoneNumber: this.params.phoneNumber }
      );

      return {
        success: response.ok,
        operation: 'get_phone_number',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
        sid: data?.sid,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_phone_number',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Buy phone number operation
   */
  private async buyPhoneNumber(): Promise<TwilioResult> {
    if (!this.params.phoneNumber) {
      throw new Error('phoneNumber is required for buy_phone_number operation');
    }

    const startTime = Date.now();

    try {
      const body = new URLSearchParams({
        PhoneNumber: this.params.phoneNumber,
      });

      if (this.params.voiceUrl) {
        body.append('VoiceUrl', this.params.voiceUrl);
      }

      const url = this.buildUrl('IncomingPhoneNumbers');
      const { response, data, timing } = await this.resilience.execute(
        `twilio-buy-phone-${this.params.phoneNumber}`,
        () => this.makeRequest('POST', url, body),
        { operation: 'buy_phone_number', phoneNumber: this.params.phoneNumber }
      );

      return {
        success: response.ok || response.status === 201,
        operation: 'buy_phone_number',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
        sid: data?.sid,
        accountSid: data?.account_sid,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'buy_phone_number',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Release phone number operation
   */
  private async releasePhoneNumber(): Promise<TwilioResult> {
    if (!this.params.phoneNumber) {
      throw new Error('phoneNumber is required for release_phone_number operation');
    }

    const startTime = Date.now();

    try {
      const url = this.buildUrl('IncomingPhoneNumbers', this.params.phoneNumber);
      const { response, data, timing } = await this.resilience.execute(
        `twilio-release-phone-${this.params.phoneNumber}`,
        () => this.makeRequest('DELETE', url),
        { operation: 'release_phone_number', phoneNumber: this.params.phoneNumber }
      );

      return {
        success: response.ok || response.status === 204,
        operation: 'release_phone_number',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'release_phone_number',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get messages operation
   */
  private async getMessages(): Promise<TwilioResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        PageSize: String(this.params.pageSize),
      });

      if (this.params.pageToken) {
        params.append('PageToken', this.params.pageToken);
      }

      const url = this.buildUrl('SMS/Messages', undefined, params.toString());
      const { response, data, timing } = await this.resilience.execute(
        'twilio-get-messages',
        () => this.makeRequest('GET', url),
        { operation: 'get_messages' }
      );

      return {
        success: response.ok,
        operation: 'get_messages',
        data: data?.messages || data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
        pagination: {
          nextPageUrl: data?.next_page_url,
          nextPageToken: data?.next_page_token,
          totalCount: data?.total,
        },
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_messages',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get account info operation
   */
  private async getAccountInfo(): Promise<TwilioResult> {
    const startTime = Date.now();

    try {
      const url = this.buildUrl('');
      const { response, data, timing } = await this.resilience.execute(
        'twilio-get-account',
        () => this.makeRequest('GET', url),
        { operation: 'get_account_info' }
      );

      return {
        success: response.ok,
        operation: 'get_account_info',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
        sid: data?.sid,
        accountSid: data?.sid,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_account_info',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get usage operation
   */
  private async getUsage(): Promise<TwilioResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        Limit: String(this.params.limit),
      });

      if (this.params.category) {
        params.append('Category', this.params.category);
      }

      if (this.params.usageCategory) {
        params.append('Category', this.params.usageCategory);
      }

      if (this.params.startDate) {
        params.append('StartDate', this.params.startDate);
      }

      if (this.params.endDate) {
        params.append('EndDate', this.params.endDate);
      }

      const url = this.buildUrl('Usage/Records', undefined, params.toString());
      const { response, data, timing } = await this.resilience.execute(
        'twilio-get-usage',
        () => this.makeRequest('GET', url),
        { operation: 'get_usage' }
      );

      return {
        success: response.ok,
        operation: 'get_usage',
        data: data?.usage_records || data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data?.error_message,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_usage',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<TwilioResult> {
    switch (this.params.operation) {
      case 'send_sms':
        return this.sendSMS();
      case 'send_bulk_sms':
        return this.sendBulkSMS();
      case 'make_call':
        return this.makeCall();
      case 'get_call_status':
        return this.getCallStatus();
      case 'record_call':
        return this.recordCall();
      case 'get_call_recording':
        return this.getCallRecording();
      case 'get_phone_number':
        return this.getPhoneNumber();
      case 'buy_phone_number':
        return this.buyPhoneNumber();
      case 'release_phone_number':
        return this.releasePhoneNumber();
      case 'get_messages':
        return this.getMessages();
      case 'get_account_info':
        return this.getAccountInfo();
      case 'get_usage':
        return this.getUsage();
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

export default TwilioBubble;
