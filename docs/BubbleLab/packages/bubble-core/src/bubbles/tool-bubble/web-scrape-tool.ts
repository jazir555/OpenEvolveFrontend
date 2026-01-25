import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * WebScrapeTool - Production web scraping with HTTP client and HTML parsing
 */
export class WebScrapeTool extends ToolBubble<WebScrapeParams, WebScrapeResult> {
  bubbleName = 'web-scrape';
  type = 'tool';
  alias = 'web-scrape';

  params = {
    timeout: z.number().int().positive().default(30000),
    maxRetries: z.number().int().positive().max(5).default(3),
    userAgent: z.string().default('Mozilla/5.0 (compatible; BubbleLab/1.0)'),
    followRedirects: z.boolean().default(true)
  };

  private userAgent: string;
  private maxRetries: number;
  private timeout: number;

  // Performance optimization: LRU cache for scraped content
  private scrapeCache = new Map<string, { data: any; timestamp: number }>();
  private readonly CACHE_TTL = 300000; // 5 minutes
  private readonly MAX_CACHE_SIZE = 200;

  // Performance: Pre-compiled regex patterns
  private static readonly TITLE_REGEX = /<title[^>]*>(.*?)<\/title>/is;
  private static readonly META_REGEX = /<meta[^>]*name=["']([^"']*)["'][^>]*content=["']([^"']*)["'][^>]*>/gi;
  private static readonly OG_REGEX = /<meta[^>]*property=["']og:([^"']*)["'][^>]*content=["']([^"']*)["'][^>]*>/gi;
  private static readonly SCRIPT_REGEX = /<script[^>]*>.*?<\/script>/gis;
  private static readonly STYLE_REGEX = /<style[^>]*>.*?<\/style>/gis;
  private static readonly TAG_REGEX = /<[^>]*>/g;

  // Performance: Rate limiting
  private requestQueue = new Map<string, number>();
  private readonly MIN_REQUEST_INTERVAL = 1000; // 1 second between requests to same domain

  // Performance: Circuit Breaker for web scraping
  private circuitBreakerState = {
    failures: 0,
    lastFailureTime: 0,
    state: 'closed' as 'closed' | 'open' | 'half-open',
    readonly FAILURE_THRESHOLD: 5,
    readonly TIMEOUT: 60000 // 60 seconds
  };

  /**
   * COMPREHENSIVE VALIDATION SCHEMAS
   * All validation rules for web scraping operations
   */

  // URL validation schema (6 rules)
  private static readonly URLSchema = z.string().max(2048).url()
    .refine(
      (url) => {
        try {
          const parsed = new URL(url);
          return ['http:', 'https:'].includes(parsed.protocol);
        } catch {
          return false;
        }
      },
      { message: 'Only HTTP/HTTPS URLs allowed' }
    )
    .refine(
      (url) => !url.includes('localhost'),
      { message: 'localhost URLs not allowed' }
    )
    .refine(
      (url) => {
        const parsed = new URL(url);
        const hostname = parsed.hostname;
        // Block private IP ranges
        return ![
          '127.', '192.168.', '10.', '172.16.', '172.31.', '169.254.'
        ].some(prefix => hostname.startsWith(prefix));
      },
      { message: 'Private IP addresses not allowed' }
    )
    .refine(
      (url) => !url.includes('file://'),
      { message: 'file:// protocol not allowed' }
    )
    .refine(
      (url) => {
        try {
          new URL(url);
          return true;
        } catch {
          return false;
        }
      },
      { message: 'Invalid URL format' }
    );

  // Credential type enum
  private static readonly CredentialType = {
    FIRECRAWL_API_KEY: 'FIRECRAWL_API_KEY',
    BASIC_AUTH: 'BASIC_AUTH',
    BEARER_TOKEN: 'BEARER_TOKEN'
  } as const;

  // Firecrawl API response schema (3 rules)
  private static readonly FirecrawlResponseSchema = z.object({
    data: z.object({
      markdown: z.string().max(1e8).optional(),
      metadata: z.object({
        title: z.string().max(256).optional(),
        statusCode: z.number().int().min(100).max(599).optional(),
        description: z.string().max(500).optional()
      }).optional()
    }),
    success: z.boolean(),
    error: z.string().max(1000).optional()
  });

  // Main web scrape parameters schema (8 rules)
  private static readonly WebScrapeParamsSchema = z.object({
    url: WebScrapeTool.URLSchema,
    timeout: z.number().int().min(1000).max(60000).default(30000),
    maxRetries: z.number().int().min(1).max(5).default(3),
    maxAge: z.number().int().min(0).max(604800000).optional(),
    format: z.enum(['markdown', 'html', 'rawHtml', 'cleaned']).default('markdown'),
    onlyMainContent: z.boolean().default(true),
    waitFor: z.number().int().min(0).max(30000).optional(),
    headers: z.record(z.string().max(4096)).max(50).optional(),
    credentials: z.record(
      z.nativeEnum(WebScrapeTool.CredentialType),
      z.string().min(1).max(4096)
    ).max(10).optional()
  });

  constructor(params: WebScrapeParams = {}) {
    super(params);
    this.userAgent = params.userAgent || this.params.userAgent.default();
    this.maxRetries = params.maxRetries || this.params.maxRetries.default();
    this.timeout = params.timeout || this.params.timeout.default();
  }

  /**
   * Performance: Clean up resources
   */
  async destroy(): Promise<void> {
    try {
      this.scrapeCache.clear();
      this.requestQueue.clear();
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  }

  /**
   * Performance: Get cached scrape result
   */
  private getCachedScrape(url: string): any | null {
    const cached = this.scrapeCache.get(url);
    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      return cached.data;
    }
    if (cached) {
      this.scrapeCache.delete(url);
    }
    return null;
  }

  /**
   * Performance: Set scrape result in cache with LRU eviction
   */
  private setCachedScrape(url: string, data: any): void {
    if (this.scrapeCache.size >= this.MAX_CACHE_SIZE) {
      const oldestKey = this.scrapeCache.keys().next().value;
      if (oldestKey) {
        this.scrapeCache.delete(oldestKey);
      }
    }
    this.scrapeCache.set(url, { data, timestamp: Date.now() });
  }

  /**
   * Performance: Rate limiting by domain
   */
  private async enforceRateLimit(url: string): Promise<void> {
    try {
      const domain = new URL(url).hostname;
      const lastRequest = this.requestQueue.get(domain) || 0;
      const now = Date.now();
      const elapsed = now - lastRequest;

      if (elapsed < this.MIN_REQUEST_INTERVAL) {
        const delay = this.MIN_REQUEST_INTERVAL - elapsed;
        await new Promise(resolve => setTimeout(resolve, delay));
      }

      this.requestQueue.set(domain, Date.now());
    } catch (error) {
      // Invalid URL, continue without rate limiting
    }
  }

  /**
   * Performance: Circuit Breaker - Check if circuit is open
   */
  private isCircuitOpen(): boolean {
    const now = Date.now();
    const timeSinceLastFailure = now - this.circuitBreakerState.lastFailureTime;

    // If circuit is open and timeout has passed, transition to half-open
    if (this.circuitBreakerState.state === 'open' && timeSinceLastFailure > this.circuitBreakerState.TIMEOUT) {
      this.circuitBreakerState.state = 'half-open';
      this.circuitBreakerState.failures = 0;
      return false;
    }

    return this.circuitBreakerState.state === 'open';
  }

  /**
   * Performance: Circuit Breaker - Record success
   */
  private recordCircuitSuccess(): void {
    this.circuitBreakerState.failures = 0;
    if (this.circuitBreakerState.state === 'half-open') {
      this.circuitBreakerState.state = 'closed';
    }
  }

  /**
   * Performance: Circuit Breaker - Record failure
   */
  private recordCircuitFailure(): void {
    this.circuitBreakerState.failures++;
    this.circuitBreakerState.lastFailureTime = Date.now();

    if (this.circuitBreakerState.failures >= this.circuitBreakerState.FAILURE_THRESHOLD) {
      this.circuitBreakerState.state = 'open';
    }
  }

  /**
   * Performance: Execute operation with circuit breaker protection
   */
  private async executeWithCircuitBreaker<T>(operation: () => Promise<T>): Promise<T> {
    // Check if circuit is open
    if (this.isCircuitOpen()) {
      throw new Error('Circuit breaker is open - too many recent failures');
    }

    try {
      const result = await operation();
      this.recordCircuitSuccess();
      return result;
    } catch (error) {
      this.recordCircuitFailure();
      throw error;
    }
  }

  async execute(input: any): Promise<WebScrapeResult> {
    // VALIDATION: Validate input against schema
    const validationResult = WebScrapeTool.WebScrapeParamsSchema.safeParse(input);
    if (!validationResult.success) {
      const errors = validationResult.error.errors.map(e =>
        `${e.path.join('.')}: ${e.message}`
      ).join('; ');
      return {
        success: false,
        error: `Validation failed: ${errors}`,
        timestamp: new Date().toISOString()
      };
    }

    const validatedInput = validationResult.data;

    try {
      const url = validatedInput.url || input.uri;
      if (!url) {
        throw new Error('URL is required');
      }

      const result = await this.scrape({
        url,
        selectors: input.selectors,
        extractMetadata: input.extractMetadata || false
      });

      return { success: true, data: result };
    } catch (error: any) {
      return { success: false, error: error.message, timestamp: new Date().toISOString() };
    }
  }

  async scrape(params: {
    url: string;
    selectors?: Record<string, string>;
    extractMetadata?: boolean;
    headers?: Record<string, string>;
  }): Promise<WebScrapeResult> {
    try {
      // Performance: Check cache first
      const cached = this.getCachedScrape(params.url);
      if (cached) {
        return { success: true, data: { ...cached, cached: true } };
      }

      // Performance: Enforce rate limiting
      await this.enforceRateLimit(params.url);

      const html = await this.fetchWithRetry(params.url, params.headers);
      const result: any = {
        url: params.url,
        html,
        timestamp: new Date().toISOString()
      };

      if (params.extractMetadata) {
        result.metadata = this.extractMetadata(html);
      }

      if (params.selectors) {
        result.extracted = this.extractSelectors(html, params.selectors);
      }

      // Performance: Cache result
      this.setCachedScrape(params.url, result);

      return { success: true, data: result };
    } catch (error: any) {
      return { success: false, error: error.message, url: params.url };
    }
  }

  async extract(params: {
    html: string;
    selectors: Record<string, string>;
  }): Promise<WebScrapeResult> {
    try {
      const extracted = this.extractSelectors(params.html, params.selectors);
      return { success: true, data: extracted };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async batch(params: {
    urls: string[];
    selectors?: Record<string, string>;
    concurrency?: number;
  }): Promise<WebScrapeResult> {
    try {
      const concurrency = params.concurrency || 3;
      const results: any[] = [];

      // Performance: Add timeout protection for batch operations
      const batchTimeout = params.urls.length * this.timeout * 2; // Allow 2x per URL
      const timeoutPromise = new Promise<WebScrapeResult>((_, reject) =>
        setTimeout(() => reject(new Error('Batch operation timeout')), batchTimeout)
      );

      const batchOperation = async () => {
        for (let i = 0; i < params.urls.length; i += concurrency) {
          const batch = params.urls.slice(i, i + concurrency);
          const batchResults = await Promise.allSettled(
            batch.map(url => this.scrape({ url, selectors: params.selectors, extractMetadata: true }))
          );

          batchResults.forEach((result, idx) => {
            if (result.status === 'fulfilled') {
              results.push(result.value);
            } else {
              results.push({
                success: false,
                url: batch[idx],
                error: result.reason?.message || 'Unknown error'
              });
            }
          });
        }

        return {
          success: true,
          data: {
            total: params.urls.length,
            successful: results.filter(r => r.success).length,
            failed: results.filter(r => !r.success).length,
            results
          }
        };
      };

      // Performance: Race between batch operation and timeout
      return await Promise.race([batchOperation(), timeoutPromise]);
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  private async fetchWithRetry(
    url: string,
    headers?: Record<string, string>,
    attempt: number = 0
  ): Promise<string> {
    // Performance: Execute with circuit breaker protection
    return this.executeWithCircuitBreaker(async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        const requestHeaders: Record<string, string> = {
          'User-Agent': this.userAgent,
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          ...headers
        };

        const response = await fetch(url, {
          method: 'GET',
          headers: requestHeaders,
          redirect: 'follow',
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
        }

        return await response.text();
      } catch (error: any) {
        if (attempt < this.maxRetries && this.isRetryableError(error)) {
          // Performance: Exponential backoff with jitter
          const baseDelay = Math.pow(2, attempt) * 1000;
          const jitter = Math.random() * 500;
          await new Promise(resolve => setTimeout(resolve, baseDelay + jitter));
          return this.fetchWithRetry(url, headers, attempt + 1);
        }
        throw error;
      }
    });
  }

  private isRetryableError(error: any): boolean {
    if (error.name === 'AbortError') return true;
    if (error.message?.includes('ECONNRESET')) return true;
    if (error.message?.includes('ETIMEDOUT')) return true;
    return true;
  }

  private extractMetadata(html: string): any {
    const metadata: any = {};

    // Performance: Use pre-compiled regex patterns
    const titleMatch = html.match(WebScrapeTool.TITLE_REGEX);
    if (titleMatch) {
      metadata.title = this.stripHTML(titleMatch[1]);
    }

    // Performance: Reset regex state before reuse
    WebScrapeTool.META_REGEX.lastIndex = 0;
    let match;
    while ((match = WebScrapeTool.META_REGEX.exec(html)) !== null) {
      metadata[match[1]] = match[2];
    }

    WebScrapeTool.OG_REGEX.lastIndex = 0;
    while ((match = WebScrapeTool.OG_REGEX.exec(html)) !== null) {
      metadata[`og:${match[1]}`] = match[2];
    }

    return metadata;
  }

  private extractSelectors(html: string, selectors: Record<string, string>): Record<string, any> {
    const result: Record<string, any> = {};

    for (const [key, selector] of Object.entries(selectors)) {
      try {
        if (selector.includes('id=')) {
          const idMatch = selector.match(/id=["']([^"']*)["']/);
          if (idMatch) {
            const regex = new RegExp(\`<[^>]*id=["']\${idMatch[1]}["'][^>]*>(.*?)</[^>]+>\`, 'is');
            const match = html.match(regex);
            if (match) {
              result[key] = this.stripHTML(match[1]);
            }
          }
        }
      } catch (error) {
        result[key] = null;
      }
    }

    return result;
  }

  private stripHTML(str: string): string {
    // Performance: Use pre-compiled regex patterns
    return str
      .replace(WebScrapeTool.SCRIPT_REGEX, '')
      .replace(WebScrapeTool.STYLE_REGEX, '')
      .replace(WebScrapeTool.TAG_REGEX, '')
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .trim();
  }
}

export interface WebScrapeParams {
  timeout?: number;
  maxRetries?: number;
  userAgent?: string;
  followRedirects?: boolean;
}

export interface WebScrapeResult {
  success: boolean;
  data?: any;
  error?: string;
  timestamp?: string;
  url?: string;
}
