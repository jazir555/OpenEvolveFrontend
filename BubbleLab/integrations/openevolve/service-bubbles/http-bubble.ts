/**
 * HTTP Client Service Bubble
 *
 * Provides generic HTTP client for making REST API calls.
 * Supports GET, POST, PUT, PATCH, DELETE operations with full resilience.
 *
 * Federation Constitution Compliant
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// HTTP-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const HttpOperationSchema = z.enum([
  'get',
  'post',
  'put',
  'patch',
  'delete',
  'head',
  'options',
]);

const HttpHeadersSchema = z.record(z.string()).optional();

const HttpAuthSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('none'),
  }),
  z.object({
    type: z.literal('basic'),
    username: z.string(),
    password: z.string(),
  }),
  z.object({
    type: z.literal('bearer'),
    token: z.string(),
  }),
  z.object({
    type: z.literal('api_key'),
    key: z.string(),
    value: z.string(),
    addTo: z.enum(['header', 'query']).default('header'),
  }),
]);

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const HttpParamsSchema = z.object({
  operation: HttpOperationSchema.describe('HTTP method to use'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  url: z.string().url().describe('Target URL (REQUIRED)'),

  // Authentication
  auth: HttpAuthSchema.default({ type: 'none' }).describe('Authentication configuration'),

  // Request configuration
  headers: HttpHeadersSchema.describe('Custom HTTP headers'),
  body: z.union([z.string(), z.record(z.unknown())]).optional().describe('Request body (JSON or raw string)'),
  queryParams: z.record(z.union([z.string(), z.number(), z.boolean()])).optional().describe('Query parameters'),

  // Timeouts and retries
  timeout: z.number().min(1000).max(120000).default(30000).describe('Request timeout in ms'),

  // Response configuration
  responseType: z.enum(['json', 'text', 'blob']).default('json').describe('Expected response type'),
  followRedirects: z.boolean().default(true).describe('Follow HTTP redirects'),
});

type HttpParamsInput = z.input<typeof HttpParamsSchema>;
type HttpParams = z.output<typeof HttpParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const HttpResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  url: z.string(),

  // Response data
  data: z.unknown().optional(),
  status: z.number(),
  statusText: z.string(),
  headers: z.record(z.string()).optional(),

  // Performance metrics
  timing: z.number().describe('Total request time in ms'),
  size: z.number().optional().describe('Response size in bytes'),

  // Error information
  error: z.string().optional(),
});

type HttpResult = z.output<typeof HttpResultSchema>;

// ============================================================================
// HTTP BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class HttpBubble extends ServiceBubble<HttpParams, HttpResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'http' as const;
  static readonly type = 'service' as const;
  static readonly schema = HttpParamsSchema;
  static readonly resultSchema = HttpResultSchema;
  static readonly credentialType = 'http_auth' as const;

  static readonly shortDescription = 'Generic HTTP client for REST API calls';
  static readonly longDescription = `
    HTTP client service bubble for making REST API requests.

    Features:
    - All HTTP methods (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS)
    - Multiple authentication types (None, Basic, Bearer, API Key)
    - Custom headers and query parameters
    - JSON and raw body support
    - Response type handling (JSON, text, blob)
    - Redirect following
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - url: Target URL (no default - must be provided)

    Federation Constitution Compliance:
    - No magic defaults (url is required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: HttpParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    HttpBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('http', DEFAULT_RESILIENCE_CONFIG);
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - url is required by schema
  }

  /**
   * Build authorization header
   */
  private buildAuthHeader(): Record<string, string> {
    const headers: Record<string, string> = {};

    switch (this.params.auth.type) {
      case 'basic':
        const credentials = Buffer.from(`${this.params.auth.username}:${this.params.auth.password}`).toString('base64');
        headers['Authorization'] = `Basic ${credentials}`;
        break;

      case 'bearer':
        headers['Authorization'] = `Bearer ${this.params.auth.token}`;
        break;

      case 'api_key':
        if (this.params.auth.addTo === 'header') {
          headers[this.params.auth.key] = this.params.auth.value;
        }
        break;

      case 'none':
      default:
        // No authentication
        break;
    }

    return headers;
  }

  /**
   * Build full URL with query parameters
   */
  private buildUrl(): string {
    let url = this.params.url;

    // Add API key to query string if configured
    if (this.params.auth.type === 'api_key' && this.params.auth.addTo === 'query') {
      const separator = url.includes('?') ? '&' : '?';
      url = `${url}${separator}${encodeURIComponent(this.params.auth.key)}=${encodeURIComponent(this.params.auth.value)}`;
    }

    // Add custom query parameters
    if (this.params.queryParams) {
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(this.params.queryParams)) {
        params.append(key, String(value));
      }

      const queryString = params.toString();
      if (queryString) {
        const separator = url.includes('?') ? '&' : '?';
        url = `${url}${separator}${queryString}`;
      }
    }

    return url;
  }

  /**
   * Build request headers
   */
  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      ...this.params.headers,
      ...this.buildAuthHeader(),
    };

    // Set Content-Type for methods that typically have a body
    if (['post', 'put', 'patch'].includes(this.params.operation)) {
      if (!headers['Content-Type']) {
        // If body is an object, assume JSON
        if (this.params.body && typeof this.params.body === 'object') {
          headers['Content-Type'] = 'application/json';
        }
      }
    }

    return headers;
  }

  /**
   * Build request body
   */
  private buildBody(): string | undefined {
    if (!this.params.body) {
      return undefined;
    }

    // If body is already a string, use it directly
    if (typeof this.params.body === 'string') {
      return this.params.body;
    }

    // Otherwise, stringify as JSON
    return JSON.stringify(this.params.body);
  }

  /**
   * Execute HTTP request with resilience
   */
  private async executeRequest(): Promise<HttpResult> {
    const startTime = Date.now();
    const url = this.buildUrl();

    try {
      const response = await this.resilience.execute(
        `http-${this.params.operation}-${url}`,
        async () => {
          return await fetch(url, {
            method: this.params.operation.toUpperCase(),
            headers: this.buildHeaders(),
            body: ['get', 'head'].includes(this.params.operation) ? undefined : this.buildBody(),
            redirect: this.params.followRedirects ? 'follow' : 'manual',
          });
        },
        { operation: this.params.operation, url }
      );

      const timing = Date.now() - startTime;

      // Parse response
      let data: unknown;
      const contentType = response.headers.get('content-type') || '';

      if (this.params.responseType === 'json' && contentType.includes('application/json')) {
        data = await response.json();
      } else if (this.params.responseType === 'text') {
        data = await response.text();
      } else if (this.params.responseType === 'blob') {
        data = await response.blob();
      } else {
        // Fallback to text
        data = await response.text();
      }

      // Extract response headers
      const headers: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        headers[key] = value;
      });

      return {
        success: response.ok,
        operation: this.params.operation,
        url: response.url,
        data,
        status: response.status,
        statusText: response.statusText,
        headers,
        timing,
        size: response.headers.get('content-length') ? parseInt(response.headers.get('content-length')!, 10) : undefined,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: this.params.operation,
        url,
        status: 0,
        statusText: 'Request failed',
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method
   */
  async action(): Promise<HttpResult> {
    return this.executeRequest();
  }
}

export default HttpBubble;
