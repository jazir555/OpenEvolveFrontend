import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * Constants for WebScrapeTool
 */
const DEFAULT_TIMEOUT_MS = 30000;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;

/**
 * Parameters for web scraping operation
 */
interface ScrapeParams {
  url: string;
  selector?: string;
  waitForSelector?: string;
  screenshot?: boolean;
  headers?: Record<string, string>;
}

/**
 * Parameters for data extraction operation
 */
interface ExtractParams {
  html: string;
  selectors: Record<string, string>;
  transform?: (data: unknown) => unknown;
}

/**
 * Parameters for batch scraping operation
 */
interface BatchParams {
  urls: string[];
  selector?: string;
  concurrency?: number;
}

/**
 * Input parameters for WebScrapeTool
 */
export interface WebScrapeParams {
  timeout?: number;
  scrape?: ScrapeParams;
  extract?: ExtractParams;
  batch?: BatchParams;
}

/**
 * Result of WebScrapeTool operation
 */
export interface WebScrapeResult {
  success: boolean;
  result?: ScrapeResult;
  error?: string;
}

/**
 * Scraped data result
 */
interface ScrapeResult {
  url?: string;
  data?: unknown;
  screenshot?: Buffer;
  metadata?: {
    title?: string;
    timestamp: string;
  };
}

/**
 * WebScrapeTool - Performs web scraping, data extraction, and batch operations
 *
 * This tool provides three main operations:
 * 1. Scrape: Scrapes web pages and extracts content
 * 2. Extract: Extracts structured data from HTML
 * 3. Batch: Performs scraping operations on multiple URLs
 *
 * All operations include proper error handling, timeout management, and retry logic.
 */
export class WebScrapeTool extends ToolBubble<WebScrapeParams, WebScrapeResult> {
  bubbleName = 'webscrape';
  type = 'tool';
  alias = 'webscrape';

  params = {
    timeout: z.number().int().positive().default(DEFAULT_TIMEOUT_MS)
  };

  /**
   * Executes the web scraping operation
   * @param input - Operation parameters
   * @returns Promise<WebScrapeResult> - Result with scraped data
   */
  async execute(input: WebScrapeParams): Promise<WebScrapeResult> {
    try {
      const result = await this.process(input);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Web scraping operation failed';
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Processes the input and routes to appropriate operation
   * @param input - Operation parameters
   * @returns Promise<ScrapeResult> - Processed result
   */
  private async process(input: WebScrapeParams): Promise<ScrapeResult> {
    if (input.scrape) {
      return await this.scrape(input.scrape);
    } else if (input.extract) {
      return await this.extract(input.extract);
    } else if (input.batch) {
      return await this.batch(input.batch);
    }
    throw new Error('No valid operation parameters provided');
  }

  /**
   * Scrapes a web page and extracts content
   * @param params - Scraping parameters including URL and selectors
   * @returns Promise<ScrapeResult> - Scraped data result
   */
  async scrape(params: ScrapeParams): Promise<ScrapeResult> {
    try {
      this.validateUrl(params.url);

      const result = await this.executeWithRetry(
        async () => await this.client.scrape(params)
      );

      return {
        url: params.url,
        data: result,
        metadata: {
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Scrape operation failed';
      throw new Error(`Failed to scrape URL: ${errorMessage}`);
    }
  }

  /**
   * Extracts structured data from HTML content
   * @param params - Extraction parameters including HTML and selectors
   * @returns Promise<ScrapeResult> - Extracted data result
   */
  async extract(params: ExtractParams): Promise<ScrapeResult> {
    try {
      if (!params.html || !params.selectors) {
        throw new Error('HTML content and selectors are required for extraction');
      }

      const result = await this.client.extract(params);

      // Apply transformation if provided
      const transformedData = params.transform
        ? params.transform(result)
        : result;

      return {
        data: transformedData,
        metadata: {
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Extract operation failed';
      throw new Error(`Failed to extract data: ${errorMessage}`);
    }
  }

  /**
   * Performs batch scraping on multiple URLs
   * @param params - Batch parameters including URLs and concurrency settings
   * @returns Promise<ScrapeResult> - Batch scraping result
   */
  async batch(params: BatchParams): Promise<ScrapeResult> {
    try {
      if (!params.urls || params.urls.length === 0) {
        throw new Error('At least one URL is required for batch scraping');
      }

      const concurrency = params.concurrency || 5;
      const results: Array<{ url: string; data?: unknown; error?: string }> = [];

      // Process URLs in batches with controlled concurrency
      for (let i = 0; i < params.urls.length; i += concurrency) {
        const batch = params.urls.slice(i, i + concurrency);
        const batchResults = await Promise.allSettled(
          batch.map(url =>
            this.scrape({ url, selector: params.selector })
              .then(result => ({ url, data: result.data }))
              .catch(error => ({
                url,
                error: error instanceof Error ? error.message : 'Unknown error'
              }))
          )
        );

        batchResults.forEach(result => {
          if (result.status === 'fulfilled') {
            results.push(result.value);
          }
        });
      }

      return {
        data: results,
        metadata: {
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Batch operation failed';
      throw new Error(`Failed to execute batch scraping: ${errorMessage}`);
    }
  }

  /**
   * Executes an operation with retry logic
   * @param operation - The operation to execute
   * @param retries - Number of retries remaining
   * @returns Promise<unknown> - Operation result
   */
  private async executeWithRetry<T>(
    operation: () => Promise<T>,
    retries: number = MAX_RETRIES
  ): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      if (retries <= 0) {
        throw error;
      }

      // Wait before retrying
      await this.delay(RETRY_DELAY_MS);

      return this.executeWithRetry(operation, retries - 1);
    }
  }

  /**
   * Validates URL format
   * @param url - URL to validate
   * @throws Error if URL is invalid
   */
  private validateUrl(url: string): void {
    try {
      new URL(url);
    } catch (error) {
      throw new Error(`Invalid URL format: ${url}`);
    }
  }

  /**
   * Delays execution for specified milliseconds
   * @param ms - Milliseconds to delay
   * @returns Promise<void>
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
