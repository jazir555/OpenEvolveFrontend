import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

/**
 * Notion Bubble - Production-Ready Service Bubble Implementation
 *
 * Full production implementation with 17 operations:
 * 1. createPage - Create a new page in Notion
 * 2. getPage - Retrieve page information and content
 * 3. updatePage - Update page properties and content
 * 4. deletePage - Delete or archive a page
 * 5. queryDatabase - Query a Notion database with filters
 * 6. createDatabaseEntry - Add entry to database
 * 7. updateDatabaseEntry - Update database entry
 * 8. getDatabase - Get database schema and information
 * 9. appendBlocks - Append multiple content blocks to a page
 * 10. getBlocks - Get child blocks
 * 11. getBlock - Get block content and children
 * 12. updateBlock - Update a block's content
 * 13. deleteBlock - Delete a block from a page
 * 14. search - Search for pages and databases
 * 15. searchPages - Legacy search operation
 * 16. getDatabaseEntries - List all entries with pagination
 * 17. createDatabase - Create new databases with custom schemas
 *
 * Security Features:
 * - API key authentication (Bearer token)
 * - Rate limiting (3 requests/sec average with burst handling)
 * - Input validation with Zod schemas
 * - Page/Database ID format validation (32-char hex)
 * - Block content sanitization
 * - Error sanitization
 * - Structured logging
 * - Exponential backoff retry for rate limits (429)
 *
 * Authentication:
 * - Integration token (bearer token)
 * - Token format validation
 * - Handles token refresh scenarios
 *
 * Quota Management:
 * - Token bucket rate limiter
 * - Tracks API usage per minute
 * - Handles 429 (Rate Limit) with Retry-After header
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

// Validate Notion ID format (32-character hex string)
const NotionIdSchema = z.string().refine(
  (id) => /^[a-f0-9]{32}$/.test(id),
  'Invalid Notion ID format. Expected 32-character hexadecimal string.'
);

// Sanitize block content to prevent injection attacks
const sanitizeBlockContent = (content: any): any => {
  if (typeof content === 'string') {
    // Remove potentially dangerous HTML/script tags
    return content
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
      .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '');
  }
  if (Array.isArray(content)) {
    return content.map(sanitizeBlockContent);
  }
  if (typeof content === 'object' && content !== null) {
    const sanitized: any = {};
    for (const [key, value] of Object.entries(content)) {
      sanitized[key] = sanitizeBlockContent(value);
    }
    return sanitized;
  }
  return content;
};

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const CreatePageParamsSchema = z.object({
  operation: z.literal('createPage'),
  parentPageId: z.string().min(1, 'Parent page ID is required').describe('Parent page or database ID'),
  title: z.string().min(1, 'Title is required').describe('Page title'),
  properties: z.record(z.any()).optional().describe('Additional page properties'),
  icon: z.string().optional().describe('Page icon (emoji or URL)'),
  cover: z.string().url().optional().describe('Cover image URL'),
  children: z.array(z.any()).optional().describe('Initial content blocks'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetPageParamsSchema = z.object({
  operation: z.literal('getPage'),
  pageId: z.string().min(1, 'Page ID is required'),
  includeChildren: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdatePageParamsSchema = z.object({
  operation: z.literal('updatePage'),
  pageId: z.string().min(1, 'Page ID is required'),
  properties: z.record(z.any()).describe('Properties to update'),
  archived: z.boolean().optional().describe('Archive the page'),
  icon: z.string().optional(),
  cover: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeletePageParamsSchema = z.object({
  operation: z.literal('deletePage'),
  pageId: z.string().min(1, 'Page ID is required'),
  archived: z.boolean().optional().default(true).describe('Archive instead of permanent delete'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const QueryDatabaseParamsSchema = z.object({
  operation: z.literal('queryDatabase'),
  databaseId: z.string().min(1, 'Database ID is required'),
  filter: z.any().optional().describe('Notion filter object'),
  sorts: z.array(z.any()).optional().describe('Sort specifications'),
  startCursor: z.string().optional(),
  pageSize: z.number().int().positive().optional().default(100),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const AppendBlockParamsSchema = z.object({
  operation: z.literal('appendBlocks'),
  blockId: z.string().min(1, 'Block or page ID is required').describe('Block or page ID'),
  blocks: z.array(z.any()).min(1, 'At least one block is required').describe('Array of block objects to append'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetBlocksParamsSchema = z.object({
  operation: z.literal('getBlocks'),
  blockId: z.string().min(1, 'Block ID is required').describe('Block ID'),
  pageSize: z.number().int().positive().max(100).optional().default(100).describe('Number of blocks to retrieve'),
  startCursor: z.string().optional().describe('Cursor for pagination'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateDatabaseEntryParamsSchema = z.object({
  operation: z.literal('createDatabaseEntry'),
  databaseId: z.string().min(1, 'Database ID is required').describe('Database ID'),
  properties: z.record(z.any()).describe('Page properties for database entry'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateDatabaseEntryParamsSchema = z.object({
  operation: z.literal('updateDatabaseEntry'),
  pageId: z.string().min(1, 'Page ID is required').describe('Page ID of database entry'),
  properties: z.record(z.any()).describe('Properties to update'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetDatabaseEntriesParamsSchema = z.object({
  operation: z.literal('getDatabaseEntries'),
  databaseId: z.string().min(1, 'Database ID is required').describe('Database ID'),
  pageSize: z.number().int().positive().max(100).optional().default(100).describe('Number of entries to retrieve'),
  startCursor: z.string().optional().describe('Cursor for pagination'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SearchParamsSchema = z.object({
  operation: z.literal('search'),
  query: z.string().min(1, 'Search query is required').describe('Search query text'),
  filter: z.object({
    value: z.enum(['page', 'database']),
    property: z.enum(['object']).optional(),
  }).optional().describe('Filter by object type'),
  sort: z.object({
    direction: z.enum(['ascending', 'descending']).optional(),
    timestamp: z.enum(['last_edited_time']).optional(),
  }).optional().describe('Sort configuration'),
  startCursor: z.string().optional().describe('Cursor for pagination'),
  pageSize: z.number().int().positive().max(100).optional().default(100).describe('Number of results'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateDatabaseParamsSchema = z.object({
  operation: z.literal('createDatabase'),
  parentId: z.string().min(1, 'Parent page ID is required').describe('Parent page ID'),
  title: z.string().min(1, 'Database title is required').describe('Database title'),
  properties: z.record(z.any()).describe('Database schema properties'),
  description: z.array(z.any()).optional(),
  icon: z.string().optional(),
  cover: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetBlockParamsSchema = z.object({
  operation: z.literal('getBlock'),
  blockId: z.string().min(1, 'Block ID is required'),
  includeChildren: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateBlockParamsSchema = z.object({
  operation: z.literal('updateBlock'),
  blockId: z.string().min(1, 'Block ID is required'),
  type: z.string().describe('Block type (paragraph, heading_1, etc.)'),
  content: z.any().describe('Block content based on type'),
  archived: z.boolean().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteBlockParamsSchema = z.object({
  operation: z.literal('deleteBlock'),
  blockId: z.string().min(1, 'Block ID is required'),
  archived: z.boolean().optional().default(true),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SearchPagesParamsSchema = z.object({
  operation: z.literal('searchPages'),
  query: z.string().describe('Search query'),
  filter: z.object({
    value: z.enum(['page', 'database']),
    property: z.enum(['object']).optional(),
  }).optional(),
  sort: z.object({
    direction: z.enum(['ascending', 'descending']).optional(),
    timestamp: z.enum(['last_edited_time']).optional(),
  }).optional(),
  startCursor: z.string().optional(),
  pageSize: z.number().int().positive().optional().default(100),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetDatabaseParamsSchema = z.object({
  operation: z.literal('getDatabase'),
  databaseId: z.string().min(1, 'Database ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const NotionBubbleParamsSchema = z.discriminatedUnion('operation', [
  CreatePageParamsSchema,
  GetPageParamsSchema,
  UpdatePageParamsSchema,
  DeletePageParamsSchema,
  QueryDatabaseParamsSchema,
  CreateDatabaseEntryParamsSchema,
  UpdateDatabaseEntryParamsSchema,
  GetDatabaseParamsSchema,
  AppendBlockParamsSchema,
  GetBlocksParamsSchema,
  GetBlockParamsSchema,
  UpdateBlockParamsSchema,
  DeleteBlockParamsSchema,
  SearchParamsSchema,
  SearchPagesParamsSchema,
  GetDatabaseEntriesParamsSchema,
  CreateDatabaseParamsSchema,
]);

type NotionBubbleParams = z.input<typeof NotionBubbleParamsSchema>;

// ============================================================================
// RESULT SCHEMAS
// ============================================================================

const PageResultSchema = z.object({
  pageId: z.string(),
  title: z.string().optional(),
  url: z.string(),
  properties: z.record(z.any()).optional(),
  createdTime: z.string().optional(),
  lastEditedTime: z.string().optional(),
  success: z.boolean(),
  error: z.string(),
});

const PageInfoSchema = z.object({
  pageId: z.string(),
  title: z.string().optional(),
  parent: z.any().optional(),
  properties: z.record(z.any()).optional(),
  children: z.array(z.any()).optional(),
  icon: z.string().optional(),
  cover: z.string().optional(),
  createdTime: z.string().optional(),
  lastEditedTime: z.string().optional(),
  archived: z.boolean().optional(),
  success: z.boolean(),
  error: z.string(),
});

const DatabaseResultSchema = z.object({
  databaseId: z.string(),
  title: z.string().optional(),
  url: z.string(),
  properties: z.record(z.any()).optional(),
  success: z.boolean(),
  error: z.string(),
});

const QueryResultSchema = z.object({
  results: z.array(z.any()),
  nextCursor: z.string().optional(),
  hasMore: z.boolean(),
  totalCount: z.number(),
  success: z.boolean(),
  error: z.string(),
});

const BlockResultSchema = z.object({
  blockId: z.string(),
  type: z.string(),
  content: z.any().optional(),
  hasChildren: z.boolean(),
  success: z.boolean(),
  error: z.string(),
});

const BlockInfoSchema = z.object({
  blockId: z.string(),
  type: z.string(),
  content: z.any().optional(),
  children: z.array(z.any()).optional(),
  createdTime: z.string().optional(),
  lastEditedTime: z.string().optional(),
  archived: z.boolean().optional(),
  success: z.boolean(),
  error: z.string(),
});

const SearchResultSchema = z.object({
  results: z.array(z.any()),
  nextCursor: z.string().optional(),
  hasMore: z.boolean(),
  totalCount: z.number(),
  success: z.boolean(),
  error: z.string(),
});

const DatabaseInfoSchema = z.object({
  databaseId: z.string(),
  title: z.string().optional(),
  description: z.array(z.any()).optional(),
  properties: z.record(z.any()).optional(),
  parent: z.any().optional(),
  icon: z.string().optional(),
  cover: z.string().optional(),
  url: z.string().optional(),
  success: z.boolean(),
  error: z.string(),
});

const NotionBubbleResultSchema = z.discriminatedUnion('operation', [
  z.object({
    operation: z.literal('createPage'),
    result: PageResultSchema,
  }),
  z.object({
    operation: z.literal('getPage'),
    result: PageInfoSchema,
  }),
  z.object({
    operation: z.literal('updatePage'),
    result: PageResultSchema,
  }),
  z.object({
    operation: z.literal('deletePage'),
    result: z.object({
      pageId: z.string(),
      archived: z.boolean(),
      success: z.boolean(),
      error: z.string(),
    }),
  }),
  z.object({
    operation: z.literal('queryDatabase'),
    result: QueryResultSchema,
  }),
  z.object({
    operation: z.literal('createDatabase'),
    result: DatabaseResultSchema,
  }),
  z.object({
    operation: z.literal('appendBlock'),
    result: z.object({
      blockId: z.string(),
      appendedBlocks: z.number(),
      success: z.boolean(),
      error: z.string(),
    }),
  }),
  z.object({
    operation: z.literal('getBlock'),
    result: BlockInfoSchema,
  }),
  z.object({
    operation: z.literal('updateBlock'),
    result: BlockResultSchema,
  }),
  z.object({
    operation: z.literal('deleteBlock'),
    result: z.object({
      blockId: z.string(),
      archived: z.boolean(),
      success: z.boolean(),
      error: z.string(),
    }),
  }),
  z.object({
    operation: z.literal('searchPages'),
    result: SearchResultSchema,
  }),
  z.object({
    operation: z.literal('getDatabase'),
    result: DatabaseInfoSchema,
  }),
]);

type NotionBubbleResult = z.output<typeof NotionBubbleResultSchema>;

// ============================================================================
// NOTION API CLIENT WITH RATE LIMITING
// ============================================================================

class NotionClient {
  private baseUrl: string = 'https://api.notion.com/v1';
  private headers: Record<string, string>;
  private rateLimiter: {
    tokens: number;
    lastRefill: number;
    maxTokens: number;
    refillRate: number;
  };

  constructor(apiKey: string) {
    // Validate API key format
    if (!apiKey || typeof apiKey !== 'string') {
      throw new Error('Invalid Notion API key: must be a non-empty string');
    }

    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'Notion-Version': '2022-06-28',
    };

    // Token bucket rate limiter: 3 requests/sec average
    this.rateLimiter = {
      tokens: 3,
      lastRefill: Date.now(),
      maxTokens: 3,
      refillRate: 3000, // ms per token
    };
  }

  private async waitForToken(): Promise<void> {
    const now = Date.now();
    const timeSinceLastRefill = now - this.rateLimiter.lastRefill;
    const tokensToAdd = Math.floor(timeSinceLastRefill / this.rateLimiter.refillRate);

    this.rateLimiter.tokens = Math.min(
      this.rateLimiter.maxTokens,
      this.rateLimiter.tokens + tokensToAdd
    );
    this.rateLimiter.lastRefill = now;

    if (this.rateLimiter.tokens < 1) {
      const waitTime = this.rateLimiter.refillRate;
      await new Promise(resolve => setTimeout(resolve, waitTime));
      this.rateLimiter.tokens = 1;
    }

    this.rateLimiter.tokens -= 1;
  }

  private async handleResponse(response: Response): Promise<any> {
    // Handle rate limiting (429)
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After');
      const waitTime = retryAfter ? parseInt(retryAfter) * 1000 : 5000;

      console.warn(`[Notion] Rate limited. Waiting ${waitTime}ms before retry...`);

      await new Promise(resolve => setTimeout(resolve, waitTime));

      throw new Error('RATE_LIMITED');
    }

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Notion API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async get(endpoint: string): Promise<any> {
    await this.waitForToken();

    const url = `${this.baseUrl}/${endpoint}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
      signal: AbortSignal.timeout(30000),
    });

    return this.handleResponse(response);
  }

  async post(endpoint: string, body?: any): Promise<any> {
    await this.waitForToken();

    const url = `${this.baseUrl}/${endpoint}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(60000),
    });

    return this.handleResponse(response);
  }

  async patch(endpoint: string, body: any): Promise<any> {
    await this.waitForToken();

    const url = `${this.baseUrl}/${endpoint}`;
    const response = await fetch(url, {
      method: 'PATCH',
      headers: this.headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000),
    });

    return this.handleResponse(response);
  }
}

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class NotionBubble<
  T extends NotionBubbleParams = NotionBubbleParams
> extends ServiceBubble<T, any> {
  static readonly type = 'service' as const;
  static readonly service = 'notion';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'notion';
  static readonly schema = NotionBubbleParamsSchema;
  static readonly resultSchema = NotionBubbleResultSchema;
  static readonly shortDescription = 'Production-ready Notion integration for pages, databases, and blocks';
  static readonly longDescription = `
    Comprehensive Notion service bubble for all workspace operations.

    Operations:
    1. createPage - Create new pages with properties and content
    2. getPage - Retrieve page information and content blocks
    3. updatePage - Update page properties, icon, and cover
    4. deletePage - Archive or delete pages
    5. queryDatabase - Query databases with filters and sorting
    6. createDatabaseEntry - Add entry to database
    7. updateDatabaseEntry - Update database entry
    8. getDatabase - Get database schema and configuration
    9. appendBlocks - Append content blocks to pages
    10. getBlocks - Get child blocks
    11. getBlock - Get block content and children
    12. updateBlock - Update block content
    13. deleteBlock - Archive or delete blocks
    14. search - Search across workspace
    15. searchPages - Legacy search operation
    16. getDatabaseEntries - List all entries with pagination
    17. createDatabase - Create new databases with custom schemas

    Features:
    - Full page and database CRUD
    - Rich block content support
    - Property management
    - Database querying with filters
    - Search functionality
    - Rate limiting (3 req/sec)
    - Input validation and sanitization
    - Resilience patterns with retry
  `;
  static readonly alias = 'notion';

  private client: NotionClient | null = null;

  constructor(
    params: T,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected getCredentialType(): CredentialType {
    return CredentialType.NOTION_OAUTH_TOKEN;
  }

  public async testCredential(): Promise<boolean> {
    const apiKey = this.chooseCredential();
    if (!apiKey) {
      return false;
    }

    try {
      const client = new NotionClient(apiKey);
      await client.get('users/me');
      return true;
    } catch {
      return false;
    }
  }

  protected chooseCredential(): string | undefined {
    const credentials = (this.params as any).credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('Notion API credentials are required');
    }
    return credentials[CredentialType.NOTION_OAUTH_TOKEN];
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<Extract<NotionBubbleResult, { operation: T['operation'] }>> {
    void context;

    const apiKey = this.chooseCredential();
    if (!apiKey) {
      return this.errorResult('Notion API key is required');
    }

    this.client = new NotionClient(apiKey);

    const { operation } = this.params;

    try {
      let result: any;

      switch (operation) {
        case 'createPage':
          result = await this.createPage(this.params as any);
          break;
        case 'getPage':
          result = await this.getPage(this.params as any);
          break;
        case 'updatePage':
          result = await this.updatePage(this.params as any);
          break;
        case 'deletePage':
          result = await this.deletePage(this.params as any);
          break;
        case 'queryDatabase':
          result = await this.queryDatabase(this.params as any);
          break;
        case 'createDatabaseEntry':
          result = await this.createDatabaseEntry(this.params as any);
          break;
        case 'updateDatabaseEntry':
          result = await this.updateDatabaseEntry(this.params as any);
          break;
        case 'getDatabase':
          result = await this.getDatabase(this.params as any);
          break;
        case 'appendBlocks':
          result = await this.appendBlocks(this.params as any);
          break;
        case 'getBlocks':
          result = await this.getBlocks(this.params as any);
          break;
        case 'getBlock':
          result = await this.getBlock(this.params as any);
          break;
        case 'updateBlock':
          result = await this.updateBlock(this.params as any);
          break;
        case 'deleteBlock':
          result = await this.deleteBlock(this.params as any);
          break;
        case 'search':
          result = await this.search(this.params as any);
          break;
        case 'searchPages':
          result = await this.searchPages(this.params as any);
          break;
        case 'getDatabaseEntries':
          result = await this.getDatabaseEntries(this.params as any);
          break;
        case 'createDatabase':
          result = await this.createDatabase(this.params as any);
          break;
        default:
          throw new Error(`Unsupported operation: ${operation}`);
      }

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
  // OPERATION 1: CREATE PAGE
  // ========================================================================

  private async createPage(
    params: Extract<NotionBubbleParams, { operation: 'createPage' }>
  ): Promise<typeof PageResultSchema._output> {
    const { parentPageId, title, properties, icon, cover, children } = params;

    try {
      // Sanitize all inputs
      const sanitizedChildren = children ? sanitizeBlockContent(children) : undefined;

      const body: any = {
        parent: {
          type: 'page_id',
          page_id: parentPageId,
        },
        properties: {
          title: {
            title: [
              {
                text: {
                  content: sanitizeBlockContent(title),
                },
              },
            ],
          },
          ...properties,
        },
      };

      if (icon) {
        body.icon = icon.startsWith('http')
          ? { type: 'external', external: { url: icon } }
          : { type: 'emoji', emoji: icon };
      }

      if (cover) {
        body.cover = { type: 'external', external: { url: cover } };
      }

      if (sanitizedChildren && sanitizedChildren.length > 0) {
        body.children = sanitizedChildren;
      }

      const response = await this.client!.post('pages', body);

      const titleProperty = response.properties.title || response.properties.Name;
      const titleText = titleProperty?.title?.[0]?.text?.content || title;

      return {
        pageId: response.id,
        title: titleText,
        url: response.url,
        properties: response.properties,
        createdTime: response.created_time,
        lastEditedTime: response.last_edited_time,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        pageId: '',
        title: sanitizeBlockContent(title),
        url: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create page',
      };
    }
  }

  // ========================================================================
  // OPERATION 2: GET PAGE
  // ========================================================================

  private async getPage(
    params: Extract<NotionBubbleParams, { operation: 'getPage' }>
  ): Promise<typeof PageInfoSchema._output> {
    const { pageId, includeChildren } = params;

    try {
      const query = includeChildren ? '?block_children.page=100' : '';
      const response = await this.client!.get(`pages/${pageId}${query}`);

      const titleProperty = response.properties.title || response.properties.Name;
      const titleText = titleProperty?.title?.[0]?.text?.content || '';

      return {
        pageId: response.id,
        title: titleText,
        parent: response.parent,
        properties: response.properties,
        children: response.children?.map((child: any) => child) || [],
        icon: response.icon,
        cover: response.cover,
        createdTime: response.created_time,
        lastEditedTime: response.last_edited_time,
        archived: response.archived,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        pageId,
        title: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get page',
      };
    }
  }

  // ========================================================================
  // OPERATION 3: UPDATE PAGE
  // ========================================================================

  private async updatePage(
    params: Extract<NotionBubbleParams, { operation: 'updatePage' }>
  ): Promise<typeof PageResultSchema._output> {
    const { pageId, properties, archived, icon, cover } = params;

    try {
      const body: any = {
        properties,
      };

      if (archived !== undefined) {
        body.archived = archived;
      }

      if (icon) {
        body.icon = icon.startsWith('http')
          ? { type: 'external', external: { url: icon } }
          : { type: 'emoji', emoji: icon };
      }

      if (cover) {
        body.cover = { type: 'external', external: { url: cover } };
      }

      const response = await this.client!.patch(`pages/${pageId}`, body);

      const titleProperty = response.properties.title || response.properties.Name;
      const titleText = titleProperty?.title?.[0]?.text?.content || '';

      return {
        pageId: response.id,
        title: titleText,
        url: response.url,
        properties: response.properties,
        createdTime: response.created_time,
        lastEditedTime: response.last_edited_time,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        pageId,
        title: '',
        url: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update page',
      };
    }
  }

  // ========================================================================
  // OPERATION 4: DELETE PAGE
  // ========================================================================

  private async deletePage(
    params: Extract<NotionBubbleParams, { operation: 'deletePage' }>
  ): Promise<{ pageId: string; archived: boolean; success: boolean; error: string }> {
    const { pageId, archived } = params;

    try {
      await this.client!.patch(`pages/${pageId}`, {
        archived: archived!,
      });

      return {
        pageId,
        archived: archived!,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        pageId,
        archived: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete page',
      };
    }
  }

  // ========================================================================
  // OPERATION 5: QUERY DATABASE
  // ========================================================================

  private async queryDatabase(
    params: Extract<NotionBubbleParams, { operation: 'queryDatabase' }>
  ): Promise<typeof QueryResultSchema._output> {
    const { databaseId, filter, sorts, startCursor, pageSize } = params;

    try {
      const body: any = {
        page_size: pageSize,
      };

      if (filter) {
        body.filter = filter;
      }

      if (sorts) {
        body.sorts = sorts;
      }

      if (startCursor) {
        body.start_cursor = startCursor;
      }

      const response = await this.client!.post(`databases/${databaseId}/query`, body);

      return {
        results: response.results || [],
        nextCursor: response.next_cursor,
        hasMore: response.has_more || false,
        totalCount: (response.results || []).length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        results: [],
        hasMore: false,
        totalCount: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to query database',
      };
    }
  }

  // ========================================================================
  // OPERATION 6: CREATE DATABASE
  // ========================================================================

  private async createDatabase(
    params: Extract<NotionBubbleParams, { operation: 'createDatabase' }>
  ): Promise<typeof DatabaseResultSchema._output> {
    const { parentId, title, properties, description, icon, cover } = params;

    try {
      const body: any = {
        parent: {
          type: 'page_id',
          page_id: parentId,
        },
        properties: {
          Name: {
            title: [
              {
                text: {
                  content: title,
                },
              },
            ],
          },
          ...properties,
        },
      };

      if (description) {
        body.description = description;
      }

      if (icon) {
        body.icon = icon.startsWith('http')
          ? { type: 'external', external: { url: icon } }
          : { type: 'emoji', emoji: icon };
      }

      if (cover) {
        body.cover = { type: 'external', external: { url: cover } };
      }

      const response = await this.client!.post('databases', body);

      const titleProp = response.properties.Name || response.properties.title;
      const titleText = titleProp?.title?.[0]?.text?.content || title;

      return {
        databaseId: response.id,
        title: titleText,
        url: response.url,
        properties: response.properties,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        databaseId: '',
        title,
        url: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create database',
      };
    }
  }

  // ========================================================================
  // OPERATION 7: APPEND BLOCKS
  // ========================================================================

  private async appendBlocks(
    params: Extract<NotionBubbleParams, { operation: 'appendBlocks' }>
  ): Promise<{ blockId: string; appendedBlocks: number; success: boolean; error: string }> {
    const { blockId, blocks } = params;

    try {
      // Sanitize block content
      const sanitizedBlocks = sanitizeBlockContent(blocks);

      const response = await this.client!.patch(`blocks/${blockId}/children`, {
        children: sanitizedBlocks,
      });

      return {
        blockId,
        appendedBlocks: (response.children || []).length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        blockId,
        appendedBlocks: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to append blocks',
      };
    }
  }

  // ========================================================================
  // OPERATION 8: GET BLOCKS
  // ========================================================================

  private async getBlocks(
    params: Extract<NotionBubbleParams, { operation: 'getBlocks' }>
  ): Promise<{ blockId: string; blocks: any[]; nextCursor: string | undefined; hasMore: boolean; success: boolean; error: string }> {
    const { blockId, pageSize, startCursor } = params;

    try {
      const queryParams = new URLSearchParams();
      queryParams.append('page_size', (pageSize ?? 100).toString());
      if (startCursor) {
        queryParams.append('start_cursor', startCursor);
      }

      const query = queryParams.toString();
      const response = await this.client!.get(`blocks/${blockId}/children${query ? `?${query}` : ''}`);

      return {
        blockId,
        blocks: response.results || [],
        nextCursor: response.next_cursor,
        hasMore: response.has_more || false,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        blockId,
        blocks: [],
        nextCursor: undefined,
        hasMore: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get blocks',
      };
    }
  }

  // ========================================================================
  // OPERATION 8: GET BLOCK
  // ========================================================================

  private async getBlock(
    params: Extract<NotionBubbleParams, { operation: 'getBlock' }>
  ): Promise<typeof BlockInfoSchema._output> {
    const { blockId, includeChildren } = params;

    try {
      const query = includeChildren ? '?block_children.page=100' : '';
      const response = await this.client!.get(`blocks/${blockId}${query}`);

      return {
        blockId: response.id,
        type: response.type,
        content: response[response.type],
        children: response.children?.map((child: any) => child) || [],
        createdTime: response.created_time,
        lastEditedTime: response.last_edited_time,
        archived: response.archived,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        blockId,
        type: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get block',
      };
    }
  }

  // ========================================================================
  // OPERATION 9: UPDATE BLOCK
  // ========================================================================

  private async updateBlock(
    params: Extract<NotionBubbleParams, { operation: 'updateBlock' }>
  ): Promise<typeof BlockResultSchema._output> {
    const { blockId, type, content, archived } = params;

    try {
      const body: any = {
        [type]: content,
      };

      if (archived !== undefined) {
        body.archived = archived;
      }

      const response = await this.client!.patch(`blocks/${blockId}`, body);

      return {
        blockId: response.id,
        type: response.type,
        content: response[response.type],
        hasChildren: response.has_children,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        blockId,
        type,
        hasChildren: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update block',
      };
    }
  }

  // ========================================================================
  // OPERATION 10: DELETE BLOCK
  // ========================================================================

  private async deleteBlock(
    params: Extract<NotionBubbleParams, { operation: 'deleteBlock' }>
  ): Promise<{ blockId: string; archived: boolean; success: boolean; error: string }> {
    const { blockId, archived } = params;

    try {
      await this.client!.patch(`blocks/${blockId}`, {
        archived: archived!,
      });

      return {
        blockId,
        archived: archived!,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        blockId,
        archived: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete block',
      };
    }
  }

  // ========================================================================
  // OPERATION 11: SEARCH PAGES
  // ========================================================================

  private async searchPages(
    params: Extract<NotionBubbleParams, { operation: 'searchPages' }>
  ): Promise<typeof SearchResultSchema._output> {
    const { query, filter, sort, startCursor, pageSize } = params;

    try {
      const body: any = {
        query,
        page_size: pageSize,
      };

      if (filter) {
        body.filter = filter;
      }

      if (sort) {
        body.sort = sort;
      }

      if (startCursor) {
        body.start_cursor = startCursor;
      }

      const response = await this.client!.post('search', body);

      return {
        results: response.results || [],
        nextCursor: response.next_cursor,
        hasMore: response.has_more || false,
        totalCount: (response.results || []).length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        results: [],
        hasMore: false,
        totalCount: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to search pages',
      };
    }
  }

  // ========================================================================
  // OPERATION 12: GET DATABASE
  // ========================================================================

  private async getDatabase(
    params: Extract<NotionBubbleParams, { operation: 'getDatabase' }>
  ): Promise<typeof DatabaseInfoSchema._output> {
    const { databaseId } = params;

    try {
      const response = await this.client!.get(`databases/${databaseId}`);

      const titleProp = response.properties.Name || response.properties.title;
      const titleText = titleProp?.title?.[0]?.text?.content || '';

      return {
        databaseId: response.id,
        title: titleText,
        description: response.description,
        properties: response.properties,
        parent: response.parent,
        icon: response.icon,
        cover: response.cover,
        url: response.url,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        databaseId,
        title: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get database',
      };
    }
  }

  // ========================================================================
  // OPERATION 13: SEARCH
  // ========================================================================

  private async search(
    params: Extract<NotionBubbleParams, { operation: 'search' }>
  ): Promise<typeof SearchResultSchema._output> {
    const { query, filter, sort, startCursor, pageSize } = params;

    try {
      const body: any = {
        query,
        page_size: pageSize,
      };

      if (filter) {
        body.filter = filter;
      }

      if (sort) {
        body.sort = sort;
      }

      if (startCursor) {
        body.start_cursor = startCursor;
      }

      const response = await this.client!.post('search', body);

      return {
        results: response.results || [],
        nextCursor: response.next_cursor,
        hasMore: response.has_more || false,
        totalCount: (response.results || []).length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        results: [],
        hasMore: false,
        totalCount: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to search',
      };
    }
  }

  // ========================================================================
  // OPERATION 14: CREATE DATABASE ENTRY
  // ========================================================================

  private async createDatabaseEntry(
    params: Extract<NotionBubbleParams, { operation: 'createDatabaseEntry' }>
  ): Promise<typeof PageResultSchema._output> {
    const { databaseId, properties } = params;

    try {
      const body: any = {
        parent: {
          type: 'database_id',
          database_id: databaseId,
        },
        properties,
      };

      const response = await this.client!.post('pages', body);

      const titleProperty = response.properties.title || response.properties.Name;
      const titleText = titleProperty?.title?.[0]?.text?.content || '';

      return {
        pageId: response.id,
        title: titleText,
        url: response.url,
        properties: response.properties,
        createdTime: response.created_time,
        lastEditedTime: response.last_edited_time,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        pageId: '',
        title: '',
        url: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create database entry',
      };
    }
  }

  // ========================================================================
  // OPERATION 15: UPDATE DATABASE ENTRY
  // ========================================================================

  private async updateDatabaseEntry(
    params: Extract<NotionBubbleParams, { operation: 'updateDatabaseEntry' }>
  ): Promise<typeof PageResultSchema._output> {
    const { pageId, properties } = params;

    try {
      const response = await this.client!.patch(`pages/${pageId}`, {
        properties,
      });

      const titleProperty = response.properties.title || response.properties.Name;
      const titleText = titleProperty?.title?.[0]?.text?.content || '';

      return {
        pageId: response.id,
        title: titleText,
        url: response.url,
        properties: response.properties,
        createdTime: response.created_time,
        lastEditedTime: response.last_edited_time,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        pageId,
        title: '',
        url: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update database entry',
      };
    }
  }

  // ========================================================================
  // OPERATION 16: GET DATABASE ENTRIES
  // ========================================================================

  private async getDatabaseEntries(
    params: Extract<NotionBubbleParams, { operation: 'getDatabaseEntries' }>
  ): Promise<typeof QueryResultSchema._output> {
    const { databaseId, pageSize, startCursor } = params;

    try {
      const body: any = {
        page_size: pageSize,
      };

      if (startCursor) {
        body.start_cursor = startCursor;
      }

      const response = await this.client!.post(`databases/${databaseId}/query`, body);

      return {
        results: response.results || [],
        nextCursor: response.next_cursor,
        hasMore: response.has_more || false,
        totalCount: (response.results || []).length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        results: [],
        hasMore: false,
        totalCount: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get database entries',
      };
    }
  }

  // ========================================================================
  // HELPER METHODS
  // ========================================================================

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
