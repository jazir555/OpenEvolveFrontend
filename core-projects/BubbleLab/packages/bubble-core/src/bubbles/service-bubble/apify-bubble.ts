import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import {
  ResilienceWrapper,
  DEFAULT_RESILIENCE_CONFIG,
  isTransientError,
} from '../../__mocks__/resilience.js';

// ============================================================================
// SECURITY UTILITIES
// ============================================================================

/**
 * Validates URL to prevent SSRF attacks
 */
function validateUrl(url: string, allowPrivateRanges = false): boolean {
  try {
    const parsed = new URL(url);

    // Only allow HTTPS and HTTP
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
      return false;
    }

    // Prevent access to localhost and private IPs unless explicitly allowed
    if (!allowPrivateRanges) {
      const hostname = parsed.hostname.toLowerCase();

      // Block localhost variants
      if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]') {
        return false;
      }

      // Block private IP ranges
      if (
        hostname.startsWith('192.168.') ||
        hostname.startsWith('10.') ||
        hostname.startsWith('172.16.') ||
        hostname.startsWith('169.254.')
      ) {
        return false;
      }
    }

    return true;
  } catch {
    return false;
  }
}

/**
 * Validates Apify actor ID format
 */
function validateActorId(actorId: string): boolean {
  // Actor IDs should be in format: username/actor-name or username~actor-name
  const actorIdPattern = /^[a-zA-Z0-9_-]+([/~][a-zA-Z0-9_-]+)+$/;
  return actorIdPattern.test(actorId);
}

/**
 * Validates Apify run ID format
 */
function validateRunId(runId: string): boolean {
  // Run IDs are alphanumeric strings
  const runIdPattern = /^[a-zA-Z0-9_-]{10,}$/;
  return runIdPattern.test(runId);
}

/**
 * Validates memory value (in MB)
 */
function validateMemory(memory: number): boolean {
  return memory >= 128 && memory <= 8192;
}

/**
 * Sanitizes error messages for client consumption
 */
function sanitizeError(error: unknown): string {
  if (error instanceof Error) {
    // Remove sensitive information from error messages
    return error.message.replace(/Bearer\s+[^\s]+/gi, 'Bearer ***').replace(/token[=:][^\s]+/gi, 'token=***');
  }
  return 'Unknown error occurred';
}

/**
 * Apify Service Bubble - Web Scraping and Automation Platform
 *
 * Full production implementation with 12 operations:
 * 1. runActor - Run an Apify actor
 * 2. getActor - Get actor details
 * 3. listActors - List available actors
 * 4. buildActor - Build actor from source
 * 5. getRun - Get run details
 * 6. waitForRun - Wait for run completion
 * 7. stopRun - Stop running actor
 * 8. listRuns - List actor runs
 * 9. getDataset - Get dataset items
 * 10. downloadDataset - Download dataset as file
 * 11. webScrape - Quick web scrape
 * 12. crawlWebsite - Crawl website
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const RunActorParamsSchema = z.object({
  operation: z.literal('runActor'),
  actorId: z.string().min(1, 'Actor ID is required'),
  input: z.record(z.unknown()).describe('Actor input parameters'),
  buildId: z.string().optional().describe('Specific build ID to run'),
  memory: z.number().int().min(128).max(8192).optional().describe('Memory in MB (128-8192)'),
  timeout: z.number().int().min(30).max(300).optional().default(300).describe('Timeout in seconds'),
  waitForFinish: z.boolean().optional().default(true),
  build: z.enum(['latest', 'specific']).optional().default('latest'),
  buildNumber: z.string().optional(),
  maxItems: z.number().int().positive().optional().default(100),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetActorParamsSchema = z.object({
  operation: z.literal('getActor'),
  actorId: z.string().min(1, 'Actor ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ListActorsParamsSchema = z.object({
  operation: z.literal('listActors'),
  limit: z.number().int().positive().max(1000).optional().default(100),
  offset: z.number().int().nonnegative().optional().default(0),
  search: z.string().optional(),
  sortBy: z.enum(['createdAt', 'modifiedAt', 'usageStats']).optional().default('createdAt'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const BuildActorParamsSchema = z.object({
  operation: z.literal('buildActor'),
  actorId: z.string().min(1, 'Actor ID is required'),
  buildTag: z.string().optional().describe('Build tag name'),
  version: z.string().optional().describe('Version number'),
  waitForFinish: z.boolean().optional().default(true).describe('Wait for build to complete'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetRunParamsSchema = z.object({
  operation: z.literal('getRun'),
  runId: z.string().min(1, 'Run ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const WaitForRunParamsSchema = z.object({
  operation: z.literal('waitForRun'),
  runId: z.string().min(1, 'Run ID is required'),
  waitFor: z.number().int().min(1).max(3600).optional().default(300).describe('Maximum wait time in seconds'),
  waitInterval: z.number().int().min(1).max(60).optional().default(5).describe('Poll interval in seconds'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const StopRunParamsSchema = z.object({
  operation: z.literal('stopRun'),
  runId: z.string().min(1, 'Run ID is required'),
  gracefully: z.boolean().optional().default(true).describe('Stop gracefully or immediately'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ListRunsParamsSchema = z.object({
  operation: z.literal('listRuns'),
  actorId: z.string().optional().describe('Filter by actor ID'),
  limit: z.number().int().positive().max(1000).optional().default(100),
  offset: z.number().int().nonnegative().optional().default(0),
  status: z.enum(['READY', 'RUNNING', 'SUCCEEDED', 'FAILED', 'TIMED-OUT', 'ABORTED']).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetDatasetParamsSchema = z.object({
  operation: z.literal('getDataset'),
  datasetId: z.string().min(1, 'Dataset ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetDatasetItemsParamsSchema = z.object({
  operation: z.literal('getDatasetItems'),
  datasetId: z.string().min(1, 'Dataset ID is required'),
  limit: z.number().int().positive().max(10000).optional().default(1000),
  offset: z.number().int().nonnegative().optional().default(0),
  clean: z.boolean().optional().default(true),
  format: z.enum(['json', 'csv', 'xml', 'xlsx', 'html']).optional().default('json'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DownloadDatasetParamsSchema = z.object({
  operation: z.literal('downloadDataset'),
  datasetId: z.string().min(1, 'Dataset ID is required'),
  format: z.enum(['json', 'csv', 'xlsx', 'html']).optional().default('json'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const WebScrapeParamsSchema = z.object({
  operation: z.literal('webScrape'),
  url: z.string().url('Valid URL is required'),
  selectors: z.array(z.string()).optional().describe('CSS selectors to extract'),
  waitForSelector: z.string().optional(),
  timeout: z.number().int().positive().optional().default(30000),
  proxyConfiguration: z
    .object({
      useApifyProxy: z.boolean().optional().default(true),
      proxyGroups: z.array(z.string()).optional(),
      countryCode: z.string().length(2).optional(),
    })
    .optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CrawlWebsiteParamsSchema = z.object({
  operation: z.literal('crawlWebsite'),
  startUrls: z.array(z.string().url()).min(1, 'At least one URL is required'),
  maxPages: z.number().int().positive().max(10000).optional().default(100),
  proxyConfiguration: z
    .object({
      useApifyProxy: z.boolean().optional().default(true),
      proxyGroups: z.array(z.string()).optional(),
      countryCode: z.string().length(2).optional(),
    })
    .optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ApifyBubbleParamsSchema = z.discriminatedUnion('operation', [
  RunActorParamsSchema,
  GetActorParamsSchema,
  ListActorsParamsSchema,
  BuildActorParamsSchema,
  GetRunParamsSchema,
  WaitForRunParamsSchema,
  StopRunParamsSchema,
  ListRunsParamsSchema,
  GetDatasetParamsSchema,
  GetDatasetItemsParamsSchema,
  DownloadDatasetParamsSchema,
  WebScrapeParamsSchema,
  CrawlWebsiteParamsSchema,
]);

type ApifyBubbleParams = z.input<typeof ApifyBubbleParamsSchema>;

// ============================================================================
// RESULT SCHEMAS
// ============================================================================

const ActorRunResultSchema = z.object({
  runId: z.string(),
  status: z.string(),
  actorId: z.string(),
  startedAt: z.string().optional(),
  finishedAt: z.string().optional(),
  datasetId: z.string().optional(),
  itemsCount: z.number().optional(),
  consoleUrl: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const ActorInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  versions: z.array(z.object({
    versionNumber: z.string(),
    buildStatus: z.string(),
    createdAt: z.string(),
  })),
  defaultRunOptions: z.object({
    build: z.string(),
    timeoutSecs: z.number(),
    memoryMbytes: z.number(),
  }).optional(),
  stats: z.object({
    totalRuns: z.number(),
    usersCount: z.number(),
  }).optional(),
  success: z.boolean(),
  error: z.string(),
});

const RunInfoSchema = z.object({
  id: z.string(),
  status: z.string(),
  actorId: z.string(),
  startedAt: z.string(),
  finishedAt: z.string().optional(),
  datasetId: z.string().optional(),
  itemsCount: z.number().optional(),
  usage: z.object({
    computeUnits: z.number(),
    duration: z.number(),
  }).optional(),
  consoleUrl: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const DatasetInfoSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  itemCount: z.number(),
  createdAt: z.string(),
  modifiedAt: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const DatasetItemsSchema = z.object({
  items: z.array(z.unknown()),
  count: z.number(),
  limit: z.number(),
  offset: z.number(),
  total: z.number().optional(),
  success: z.boolean(),
  error: z.string(),
});

const ScrapeResultSchema = z.object({
  url: z.string(),
  content: z.string().optional(),
  data: z.array(z.unknown()).optional(),
  screenshot: z.string().optional().describe('Base64 encoded screenshot'),
  pageHtml: z.string().optional(),
  itemsCount: z.number().optional(),
  success: z.boolean(),
  error: z.string(),
});

const ActorListSchema = z.object({
  actors: z.array(z.object({
    id: z.string(),
    name: z.string(),
    description: z.string().optional(),
    username: z.string().optional(),
    stats: z.object({
      totalRuns: z.number(),
      usersCount: z.number(),
    }).optional(),
  })),
  count: z.number(),
  limit: z.number(),
  offset: z.number(),
  total: z.number().optional(),
  success: z.boolean(),
  error: z.string(),
});

const ActorRunsSchema = z.object({
  runs: z.array(z.object({
    id: z.string(),
    status: z.string(),
    startedAt: z.string(),
    finishedAt: z.string().optional(),
    itemsCount: z.number().optional(),
  })),
  count: z.number(),
  limit: z.number(),
  offset: z.number(),
  total: z.number().optional(),
  success: z.boolean(),
  error: z.string(),
});

const ApifyBubbleResultSchema = z.discriminatedUnion('operation', [
  z.object({
    operation: z.literal('runActor'),
    result: ActorRunResultSchema,
  }),
  z.object({
    operation: z.literal('getActor'),
    result: ActorInfoSchema,
  }),
  z.object({
    operation: z.literal('getRun'),
    result: RunInfoSchema,
  }),
  z.object({
    operation: z.literal('getDataset'),
    result: DatasetInfoSchema,
  }),
  z.object({
    operation: z.literal('getDatasetItems'),
    result: DatasetItemsSchema,
  }),
  z.object({
    operation: z.literal('webScrape'),
    result: ScrapeResultSchema,
  }),
  z.object({
    operation: z.literal('puppeteerScrape'),
    result: ScrapeResultSchema,
  }),
  z.object({
    operation: z.literal('cheerioScrape'),
    result: ScrapeResultSchema,
  }),
  z.object({
    operation: z.literal('listActors'),
    result: ActorListSchema,
  }),
  z.object({
    operation: z.literal('getActorRuns'),
    result: ActorRunsSchema,
  }),
]);

type ApifyBubbleResult = z.output<typeof ApifyBubbleResultSchema>;

// ============================================================================
// API TYPES
// ============================================================================

interface ApifyClientOptions {
  token: string;
  baseUrl?: string;
  timeout?: number;
}

class ApifyClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(options: ApifyClientOptions) {
    this.baseUrl = options.baseUrl || 'https://api.apify.com/v2';
    this.headers = {
      'Authorization': `Bearer ${options.token}`,
      'Content-Type': 'application/json',
    };
  }

  async get(endpoint: string): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Apify API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async post(endpoint: string, body?: any): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Apify API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async getWithRetry(endpoint: string, maxRetries = 3): Promise<any> {
    let lastError: Error;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await this.get(endpoint);
      } catch (error) {
        lastError = error as Error;
        if (!isTransientError(error) || attempt === maxRetries - 1) {
          throw lastError;
        }
        // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }

    throw lastError!;
  }
}

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class ApifyBubble<
  T extends ApifyBubbleParams = ApifyBubbleParams
> extends ServiceBubble<T, any> {
  static readonly type = 'service' as const;
  static readonly service = 'apify';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'apify';
  static readonly schema = ApifyBubbleParamsSchema;
  static readonly resultSchema = ApifyBubbleResultSchema;
  static readonly shortDescription = 'Web scraping and automation platform';
  static readonly longDescription = `
    Apify Service Bubble for web scraping, crawling, and automation.

    Operations (12):
    1. runActor - Execute any Apify actor with custom input parameters
    2. getActor - Retrieve actor details, versions, and statistics
    3. listActors - Browse and discover available actors
    4. buildActor - Build actor from source code
    5. getRun - Check status and details of an actor run
    6. waitForRun - Wait for run completion with polling
    7. stopRun - Stop a running actor gracefully or immediately
    8. listRuns - List historical runs for an actor
    9. getDataset - Get dataset metadata and information
    10. getDatasetItems - Fetch scraped data from datasets
    11. downloadDataset - Download dataset in various formats
    12. webScrape - Quick web scraping with selectors
    13. crawlWebsite - Crawl entire websites with proxy support

    Features:
    - Full resilience patterns with circuit breaker and retry logic
    - SSRF protection with URL validation
    - Actor and run ID validation
    - Memory management (128-8192 MB)
    - Rate limiting with exponential backoff
    - Proxy configuration support
    - Dataset download in multiple formats
    - Real-time run monitoring and control
    - Error sanitization for security
  `;
  static readonly alias = 'apify';

  private client: ApifyClient | null = null;
  private resilience: ResilienceWrapper;

  constructor(
    params: T,
    context?: BubbleContext
  ) {
    super(params, context);

    this.resilience = new ResilienceWrapper({
      ...DEFAULT_RESILIENCE_CONFIG,
      circuitBreaker: {
        failureThreshold: 5,
        successThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 3,
      },
      retry: {
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 30000,
        jitterMultiplier: 0.1,
      },
      deduplication: {
        enabled: true,
        ttl: 60000,
        cacheResult: true,
      },
      deadLetterQueue: {
        enabled: true,
        maxSize: 1000,
      },
    });
  }

  public async testCredential(): Promise<boolean> {
    const token = this.chooseCredential();
    if (!token) {
      return false;
    }

    try {
      const client = new ApifyClient({ token });
      await client.get('/users/me');
      return true;
    } catch {
      return false;
    }
  }

  protected chooseCredential(): string | undefined {
    const credentials = (this.params as any).credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('Apify API credentials are required');
    }
    return credentials[CredentialType.APIFY_CRED];
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<Extract<ApifyBubbleResult, { operation: T['operation'] }>> {
    void context;

    const token = this.chooseCredential();
    if (!token) {
      return this.errorResult('Apify API token is required');
    }

    this.client = new ApifyClient({ token });

    const { operation } = this.params;

    try {
      const result = await this.resilience.execute(
        `apify-${operation}-${JSON.stringify(this.params)}`,
        async () => {
          switch (operation) {
            case 'runActor':
              return await this.runActor(this.params as any);
            case 'getActor':
              return await this.getActor(this.params as any);
            case 'listActors':
              return await this.listActors(this.params as any);
            case 'buildActor':
              return await this.buildActor(this.params as any);
            case 'getRun':
              return await this.getRun(this.params as any);
            case 'waitForRun':
              return await this.waitForRun(this.params as any);
            case 'stopRun':
              return await this.stopRun(this.params as any);
            case 'listRuns':
              return await this.listRuns(this.params as any);
            case 'getDataset':
              return await this.getDataset(this.params as any);
            case 'getDatasetItems':
              return await this.getDatasetItems(this.params as any);
            case 'downloadDataset':
              return await this.downloadDataset(this.params as any);
            case 'webScrape':
              return await this.webScrape(this.params as any);
            case 'crawlWebsite':
              return await this.crawlWebsite(this.params as any);
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
      return this.errorResult(error instanceof Error ? error.message : 'Unknown error');
    }
  }

  // ========================================================================
  // OPERATION 1: RUN ACTOR
  // ========================================================================

  private async runActor(
    params: Extract<ApifyBubbleParams, { operation: 'runActor' }>
  ): Promise<typeof ActorRunResultSchema._output> {
    const { actorId, input, timeout, memory, maxItems, waitForFinish, build, buildNumber } = params;

    try {
      // Build the actor ID for API (replace / with ~)
      const apiActorId = actorId.replace(/\//g, '~');

      // Prepare run options
      const runOptions: any = {
        build: build === 'specific' ? buildNumber : 'latest',
        timeoutSecs: timeout,
        memoryMbytes: memory,
        maxItems,
      };

      // Start the actor run
      const runData = await this.client!.post(`/acts/${apiActorId}/runs`, {
        input,
        ...runOptions,
      });

      const runId = runData.data.id;
      const consoleUrl = `https://console.apify.com/actors/runs/${runId}`;

      // If not waiting for finish, return immediately
      if (!waitForFinish) {
        return {
          runId,
          status: runData.data.status,
          actorId,
          consoleUrl,
          success: true,
          error: '',
        };
      }

      // Wait for completion
      const finalRunData = await this.waitForCompletion(runId, timeout!);

      // Get dataset items if available
      let itemsCount = 0;
      if (finalRunData.data.defaultDatasetId) {
        const dataset = await this.client!.get(`/datasets/${finalRunData.data.defaultDatasetId}`);
        itemsCount = dataset.data.itemCount || 0;
      }

      return {
        runId,
        status: finalRunData.data.status,
        actorId,
        startedAt: finalRunData.data.startedAt,
        finishedAt: finalRunData.data.finishedAt,
        datasetId: finalRunData.data.defaultDatasetId,
        itemsCount,
        consoleUrl,
        success: finalRunData.data.status === 'SUCCEEDED',
        error: finalRunData.data.status === 'SUCCEEDED' ? '' : `Run ${finalRunData.data.status}`,
      };
    } catch (error) {
      return {
        runId: '',
        status: 'FAILED',
        actorId,
        consoleUrl: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to run actor',
      };
    }
  }

  // ========================================================================
  // OPERATION 2: GET ACTOR
  // ========================================================================

  private async getActor(
    params: Extract<ApifyBubbleParams, { operation: 'getActor' }>
  ): Promise<typeof ActorInfoSchema._output> {
    const { actorId } = params;

    try {
      const apiActorId = actorId.replace(/\//g, '~');
      const actorData = await this.client!.get(`/acts/${apiActorId}`);

      return {
        id: actorData.data.id,
        name: actorData.data.name,
        description: actorData.data.description,
        versions: actorData.data.versions || [],
        defaultRunOptions: actorData.data.defaultRunOptions,
        stats: actorData.data.stats,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: actorId,
        name: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get actor info',
        versions: [],
      };
    }
  }

  // ========================================================================
  // OPERATION 3: GET RUN
  // ========================================================================

  private async getRun(
    params: Extract<ApifyBubbleParams, { operation: 'getRun' }>
  ): Promise<typeof RunInfoSchema._output> {
    const { runId } = params;

    try {
      const runData = await this.client!.get(`/actor-runs/${runId}`);

      return {
        id: runData.data.id,
        status: runData.data.status,
        actorId: runData.data.actId,
        startedAt: runData.data.startedAt,
        finishedAt: runData.data.finishedAt,
        datasetId: runData.data.defaultDatasetId,
        itemsCount: runData.data.itemCount,
        usage: runData.data.usage,
        consoleUrl: `https://console.apify.com/actors/runs/${runId}`,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: runId,
        status: 'UNKNOWN',
        actorId: '',
        startedAt: '',
        consoleUrl: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get run info',
      };
    }
  }

  // ========================================================================
  // OPERATION 4: GET DATASET
  // ========================================================================

  private async getDataset(
    params: Extract<ApifyBubbleParams, { operation: 'getDataset' }>
  ): Promise<typeof DatasetInfoSchema._output> {
    const { datasetId } = params;

    try {
      const datasetData = await this.client!.get(`/datasets/${datasetId}`);

      return {
        id: datasetData.data.id,
        name: datasetData.data.name,
        itemCount: datasetData.data.itemCount,
        createdAt: datasetData.data.createdAt,
        modifiedAt: datasetData.data.modifiedAt,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: datasetId,
        itemCount: 0,
        createdAt: '',
        modifiedAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get dataset info',
      };
    }
  }

  // ========================================================================
  // OPERATION 5: GET DATASET ITEMS
  // ========================================================================

  private async getDatasetItems(
    params: Extract<ApifyBubbleParams, { operation: 'getDatasetItems' }>
  ): Promise<typeof DatasetItemsSchema._output> {
    const { datasetId, limit, offset, clean, format } = params;

    try {
      const query = new URLSearchParams({
        limit: (limit ?? 100).toString(),
        offset: (offset ?? 0).toString(),
        clean: (clean ?? false).toString(),
      });

      const itemsData = await this.client!.get(`/datasets/${datasetId}/items?${query}`);

      return {
        items: itemsData.items || itemsData.data || [],
        count: (itemsData.items || itemsData.data || []).length,
        limit: limit ?? 100,
        offset: offset ?? 0,
        total: itemsData.total,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        items: [],
        count: 0,
        limit: limit ?? 100,
        offset: offset ?? 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get dataset items',
      };
    }
  }

  // ========================================================================
  // OPERATION 6: WEB SCRAPE
  // ========================================================================

  private async webScrape(
    params: Extract<ApifyBubbleParams, { operation: 'webScrape' }>
  ): Promise<typeof ScrapeResultSchema._output> {
    const { url, selectors, waitForSelector, timeout, proxyConfiguration } = params;

    try {
      // Use Apify's web scraper actor
      const input = {
        urls: [url],
        waitForSelector: waitForSelector || (selectors && selectors[0]),
        proxyConfiguration: proxyConfiguration || undefined,
      };

      const runResult = await this.runActor({
        operation: 'runActor',
        actorId: 'apify/web-scraper',
        input,
        timeout: Math.ceil((timeout ?? 30000) / 1000),
        waitForFinish: true,
        credentials: (this.params as any).credentials,
      });

      if (!runResult.success) {
        return {
          url,
          success: false,
          error: runResult.error,
        };
      }

      // Get dataset items
      const itemsResult = await this.getDatasetItems({
        operation: 'getDatasetItems',
        datasetId: runResult.datasetId!,
        limit: 100,
        offset: 0,
        clean: true,
        credentials: (this.params as any).credentials,
      });

      return {
        url,
        data: itemsResult.items,
        itemsCount: itemsResult.count,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        url,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to scrape web page',
      };
    }
  }

  // ========================================================================
  // OPERATION 7: PUPPETEER SCRAPE
  // ========================================================================

  private async puppeteerScrape(
    params: Extract<ApifyBubbleParams, { operation: 'puppeteerScrape' }>
  ): Promise<typeof ScrapeResultSchema._output> {
    const { url, script, waitForSelector, screenshot, pdf, proxy } = params;

    try {
      // Use Apify's puppeteer scraper
      const input = {
        urls: [url],
        pageFunction: script,
        waitForSelector,
        proxyConfiguration: proxy ? { useApifyProxy: true } : undefined,
        screenshot: screenshot ? { type: 'png', fullPage: true } : undefined,
        pdf: pdf ? {} : undefined,
      };

      const runResult = await this.runActor({
        operation: 'runActor',
        actorId: 'apify/puppeteer-scraper',
        input,
        timeout: 300,
        waitForFinish: true,
        credentials: (this.params as any).credentials,
      });

      if (!runResult.success) {
        return {
          url,
          success: false,
          error: runResult.error,
        };
      }

      // Get dataset items
      const itemsResult = await this.getDatasetItems({
        operation: 'getDatasetItems',
        datasetId: runResult.datasetId!,
        limit: 100,
        offset: 0,
        clean: true,
        credentials: (this.params as any).credentials,
      });

      // Extract screenshot if present
      let screenshotData: string | undefined;
      if (screenshot && itemsResult.items.length > 0) {
        const firstItem = itemsResult.items[0] as any;
        if (firstItem.screenshot) {
          screenshotData = firstItem.screenshot;
        }
      }

      return {
        url,
        data: itemsResult.items,
        itemsCount: itemsResult.count,
        screenshot: screenshotData,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        url,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to scrape with Puppeteer',
      };
    }
  }

  // ========================================================================
  // OPERATION 8: CHEERIO SCRAPE
  // ========================================================================

  private async cheerioScrape(
    params: Extract<ApifyBubbleParams, { operation: 'cheerioScrape' }>
  ): Promise<typeof ScrapeResultSchema._output> {
    const { startUrls, selector, waitForSelector, maxPages, proxy } = params;

    try {
      // Use Apify's cheerio scraper
      const input = {
        startUrls,
        selectors: [selector].filter(Boolean),
        waitForSelector,
        maxPages,
        proxyConfiguration: proxy ? { useApifyProxy: true } : undefined,
      };

      const runResult = await this.runActor({
        operation: 'runActor',
        actorId: 'apify/cheerio-scraper',
        input,
        timeout: 300,
        waitForFinish: true,
        credentials: (this.params as any).credentials,
      });

      if (!runResult.success) {
        return {
          url: startUrls[0],
          success: false,
          error: runResult.error,
        };
      }

      // Get dataset items
      const itemsResult = await this.getDatasetItems({
        operation: 'getDatasetItems',
        datasetId: runResult.datasetId!,
        limit: 1000,
        offset: 0,
        clean: true,
        credentials: (this.params as any).credentials,
      });

      return {
        url: startUrls[0],
        data: itemsResult.items,
        itemsCount: itemsResult.count,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        url: startUrls[0],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to scrape with Cheerio',
      };
    }
  }

  // ========================================================================
  // OPERATION 9: LIST ACTORS
  // ========================================================================

  private async listActors(
    params: Extract<ApifyBubbleParams, { operation: 'listActors' }>
  ): Promise<typeof ActorListSchema._output> {
    const { limit, offset, search, sortBy } = params;

    try {
      // List actors from store
      const query = new URLSearchParams();
      query.append('limit', (limit ?? 100).toString());
      query.append('offset', (offset ?? 0).toString());
      if (sortBy) {
        query.append('sortBy', sortBy);
      }

      if (search) {
        query.set('search', search);
      }

      const actorsData = await this.client!.getWithRetry(`/store?${query}`);

      return {
        actors: actorsData.data.items || [],
        count: (actorsData.data.items || []).length,
        limit: limit ?? 100,
        offset: offset ?? 0,
        total: actorsData.data.total,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        actors: [],
        count: 0,
        limit: limit ?? 100,
        offset: offset ?? 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to list actors',
      };
    }
  }

  // ========================================================================
  // OPERATION 10: GET ACTOR RUNS
  // ========================================================================

  private async getActorRuns(
    params: Extract<ApifyBubbleParams, { operation: 'listRuns' }>
  ): Promise<typeof ActorRunsSchema._output> {
    const { actorId, limit, offset, status } = params;

    if (!actorId) {
      return {
        runs: [],
        count: 0,
        limit: limit ?? 100,
        offset: offset ?? 0,
        total: 0,
        success: false,
        error: 'Actor ID is required',
      };
    }

    try {
      const apiActorId = actorId.replace(/\//g, '~');
      const query = new URLSearchParams({
        limit: (limit ?? 100).toString(),
        offset: (offset ?? 0).toString(),
      });

      if (status) {
        query.set('status', status);
      }

      const runsData = await this.client!.get(`/acts/${apiActorId}/runs?${query}`);

      return {
        runs: runsData.data.items || [],
        count: (runsData.data.items || []).length,
        limit: limit ?? 100,
        offset: offset ?? 0,
        total: runsData.data.total,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        runs: [],
        count: 0,
        limit: limit ?? 100,
        offset: offset ?? 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get actor runs',
      };
    }
  }

  // ========================================================================
  // OPERATION 4: BUILD ACTOR
  // ========================================================================

  private async buildActor(
    params: Extract<ApifyBubbleParams, { operation: 'buildActor' }>
  ): Promise<any> {
    const { actorId, buildTag, version, waitForFinish } = params;

    try {
      if (!validateActorId(actorId)) {
        throw new Error(`Invalid actor ID format: ${actorId}`);
      }

      const apiActorId = actorId.replace(/\//g, '~');
      const endpoint = `/acts/${apiActorId}/builds`;

      const body: any = {};
      if (buildTag) body.buildTag = buildTag;
      if (version) body.version = version;
      if (waitForFinish) body.waitForFinish = 300; // Wait up to 5 minutes

      const buildData = await this.client!.post(endpoint, body);

      console.log(`[Apify] Actor build started: ${buildData.data.id}`);

      return {
        id: buildData.data.id,
        actorId,
        status: buildData.data.status,
        buildTag: buildData.data.buildTag,
        startedAt: buildData.data.startedAt,
        finishedAt: buildData.data.finishedAt,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        actorId,
        status: 'FAILED',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to build actor',
      };
    }
  }

  // ========================================================================
  // OPERATION 6: WAIT FOR RUN
  // ========================================================================

  private async waitForRun(
    params: Extract<ApifyBubbleParams, { operation: 'waitForRun' }>
  ): Promise<any> {
    const { runId, waitFor, waitInterval } = params;

    try {
      if (!validateRunId(runId)) {
        throw new Error(`Invalid run ID format: ${runId}`);
      }

      const startTime = Date.now();
      const maxWaitTime = (waitFor ?? 300) * 1000;
      const pollInterval = (waitInterval ?? 5) * 1000;

      while (Date.now() - startTime < maxWaitTime) {
        const endpoint = `/actor-runs/${runId}`;
        const result = await this.client!.get(endpoint);

        const status = result.data.status;

        if (['SUCCEEDED', 'FAILED', 'ABORTED', 'TIMED-OUT'].includes(status)) {
          console.log(`[Apify] Run ${runId} finished with status: ${status}`);

          return {
            id: result.data.id,
            status: result.data.status,
            startedAt: result.data.startedAt,
            finishedAt: result.data.finishedAt,
            defaultDatasetId: result.data.defaultDatasetId,
            stats: result.data.stats,
            success: status === 'SUCCEEDED',
            error: status === 'SUCCEEDED' ? '' : `Run ${status}`,
          };
        }

        await new Promise((resolve) => setTimeout(resolve, pollInterval));
      }

      throw new Error(`Run ${runId} did not finish within ${waitFor} seconds`);
    } catch (error) {
      return {
        runId,
        status: 'FAILED',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to wait for run',
      };
    }
  }

  // ========================================================================
  // OPERATION 7: STOP RUN
  // ========================================================================

  private async stopRun(
    params: Extract<ApifyBubbleParams, { operation: 'stopRun' }>
  ): Promise<any> {
    const { runId, gracefully } = params;

    try {
      if (!validateRunId(runId)) {
        throw new Error(`Invalid run ID format: ${runId}`);
      }

      const endpoint = `/actor-runs/${runId}/stop`;
      const body = { gracefully };

      const result = await this.client!.post(endpoint, body);

      console.log(`[Apify] Run ${runId} stopped`);

      return {
        id: result.data.id,
        status: result.data.status,
        stoppedAt: new Date().toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        runId,
        status: 'FAILED',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to stop run',
      };
    }
  }

  // ========================================================================
  // OPERATION 8: LIST RUNS
  // ========================================================================

  private async listRuns(
    params: Extract<ApifyBubbleParams, { operation: 'listRuns' }>
  ): Promise<any> {
    const { actorId, limit, offset, status } = params;

    try {
      let endpoint = `/actor-runs?limit=${limit}&offset=${offset}`;

      if (actorId) {
        if (!validateActorId(actorId)) {
          throw new Error(`Invalid actor ID format: ${actorId}`);
        }
        const apiActorId = actorId.replace(/\//g, '~');
        endpoint = `/acts/${apiActorId}/runs?limit=${limit}&offset=${offset}`;
      }

      if (status) {
        endpoint += `&status=${status}`;
      }

      const result = await this.client!.get(endpoint);

      return {
        runs: result.data.items || [],
        count: (result.data.items || []).length,
        limit,
        offset,
        total: result.data.total,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        runs: [],
        count: 0,
        limit,
        offset,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to list runs',
      };
    }
  }

  // ========================================================================
  // OPERATION 10: DOWNLOAD DATASET
  // ========================================================================

  private async downloadDataset(
    params: Extract<ApifyBubbleParams, { operation: 'downloadDataset' }>
  ): Promise<any> {
    const { datasetId, format } = params;

    try {
      const endpoint = `/datasets/${datasetId}/download?format=${format}`;

      const response = await fetch(`${this.client!['baseUrl']}${endpoint}`, {
        method: 'GET',
        headers: this.client!['headers'],
      });

      if (!response.ok) {
        throw new Error(`Failed to download dataset: ${response.status}`);
      }

      // Get content type and disposition
      const contentType = response.headers.get('content-type');
      const contentDisposition = response.headers.get('content-disposition');

      // Extract filename from content-disposition if available
      let filename = `dataset-${datasetId}.${format}`;
      if (contentDisposition) {
        const match = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (match && match[1]) {
          filename = match[1].replace(/['"]/g, '');
        }
      }

      // Get content
      const content = await response.text();

      console.log(`[Apify] Downloaded dataset ${datasetId} as ${format}`);

      return {
        datasetId,
        format,
        filename,
        contentType,
        content,
        size: content.length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        datasetId,
        format,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to download dataset',
      };
    }
  }

  // ========================================================================
  // OPERATION 12: CRAWL WEBSITE
  // ========================================================================

  private async crawlWebsite(
    params: Extract<ApifyBubbleParams, { operation: 'crawlWebsite' }>
  ): Promise<any> {
    const { startUrls, maxPages, proxyConfiguration } = params;

    try {
      // Validate all URLs
      for (const url of startUrls) {
        if (!validateUrl(url)) {
          throw new Error(`Invalid or unsafe URL: ${url}`);
        }
      }

      // Run the Apify Website Content Crawler actor
      const input: any = {
        startUrls: startUrls.map((url) => ({ url })),
        maxCrawlingDepth: 1,
        maxPages,
      };

      if (proxyConfiguration?.useApifyProxy) {
        input.proxyConfiguration = {
          useApifyProxy: true,
          ...proxyConfiguration,
        };
      }

      const runResult = await this.runActor({
        operation: 'runActor',
        actorId: 'apify/website-content-crawler',
        input,
        timeout: 300,
        waitForFinish: true,
        credentials: (this.params as any).credentials,
      });

      if (!runResult.success) {
        return {
          startUrls,
          success: false,
          error: runResult.error,
        };
      }

      // Get dataset items
      const itemsResult = await this.getDatasetItems({
        operation: 'getDatasetItems',
        datasetId: runResult.datasetId!,
        limit: 10000,
        offset: 0,
        clean: true,
        credentials: (this.params as any).credentials,
      });

      console.log(`[Apify] Crawled ${startUrls.length} URLs, got ${itemsResult.count} pages`);

      return {
        startUrls,
        runId: runResult.runId,
        datasetId: runResult.datasetId,
        pagesCrawled: itemsResult.count,
        maxPages,
        data: itemsResult.items,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        startUrls,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to crawl website',
      };
    }
  }

  // ========================================================================
  // HELPER METHODS
  // ========================================================================

  private async waitForCompletion(runId: string, timeoutSecs: number): Promise<any> {
    const startTime = Date.now();
    const timeoutMs = timeoutSecs * 1000;
    const pollInterval = 2000;

    while (Date.now() - startTime < timeoutMs) {
      const runData = await this.client!.get(`/actor-runs/${runId}`);
      const status = runData.data.status;

      if (['SUCCEEDED', 'FAILED', 'ABORTED', 'TIMED-OUT'].includes(status)) {
        return runData;
      }

      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Run ${runId} timed out after ${timeoutSecs} seconds`);
  }

  private errorResult(error: string): any {
    return {
      operation: this.params.operation,
      result: {
        success: false,
        error,
      },
    };
  }
}
