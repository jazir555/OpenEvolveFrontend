/**
 * Test Suite for NotionBubble (notion-bubble.ts)
 *
 * The previous contents of this file were a generated placeholder that tested a
 * non-existent API (`instance.authenticate()`, `instance.execute()`, an `env`
 * context) and did not even parse. It has been rewritten against the real
 * bubble:
 *  - discriminated-union params keyed on camelCase `operation`
 *    (createPage / getPage / updatePage / deletePage / queryDatabase /
 *     createDatabase / createDatabaseEntry / updateDatabaseEntry /
 *     getDatabaseEntries / appendBlocks / getBlocks / getBlock / updateBlock /
 *     deleteBlock / search / searchPages / getDatabase)
 *  - `performAction()` returns `{ operation, result }` — the per-operation
 *    payload lives under `result`
 *  - schema failures are captured at construction (NOT thrown) and surfaced as a
 *    controlled error from `action()`
 *  - credentials use CredentialType.NOTION_OAUTH_TOKEN
 *  - all network access goes through the global `fetch`, mocked here
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { NotionBubble } from './notion-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

const mockCredentials = {
  [CredentialType.NOTION_OAUTH_TOKEN]: 'secret_test_notion_token',
};

function jsonResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

function errorResponse(status: number, text: string) {
  return {
    ok: false,
    status,
    statusText: 'Error',
    headers: new Headers(),
    json: async () => ({ message: text }),
    text: async () => text,
  } as unknown as Response;
}

const PAGE_RESPONSE = {
  id: 'page_123',
  url: 'https://notion.so/page_123',
  created_time: '2024-01-01T00:00:00.000Z',
  last_edited_time: '2024-01-01T01:00:00.000Z',
  archived: false,
  parent: { type: 'page_id', page_id: 'parent_1' },
  properties: {
    title: { title: [{ text: { content: 'Test Page' } }] },
  },
};

describe('NotionBubble (notion-bubble)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ========================================
  // METADATA
  // ========================================
  describe('Bubble metadata', () => {
    it('exposes the expected static metadata', () => {
      expect(NotionBubble.bubbleName).toBe('notion');
      expect(NotionBubble.type).toBe('service');
      expect(NotionBubble.service).toBe('notion');
      expect(NotionBubble.authType).toBe('apikey');
      expect(NotionBubble.alias).toBe('notion');
      expect(NotionBubble.schema).toBeDefined();
      expect(NotionBubble.resultSchema).toBeDefined();
    });

    it('excludes credentials from currentParams', () => {
      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        credentials: mockCredentials,
      });
      expect(
        (bubble.currentParams as Record<string, unknown>).credentials
      ).toBeUndefined();
    });

    it('applies schema defaults', () => {
      const bubble = new NotionBubble({
        operation: 'queryDatabase',
        databaseId: 'db_123',
        credentials: mockCredentials,
      });
      expect(
        (bubble.currentParams as Record<string, unknown>).pageSize
      ).toBe(100);
    });
  });

  // ========================================
  // INPUT VALIDATION
  // ========================================
  // The base constructor records validation failures instead of throwing.
  describe('Input Validation', () => {
    it('returns a controlled error for an unknown operation', async () => {
      const bubble = new NotionBubble({
        // @ts-expect-error deliberately invalid operation
        operation: 'explodeWorkspace',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('requires a page ID for getPage', async () => {
      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: '',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Page ID is required');
    });

    it('requires a title for createPage', async () => {
      const bubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_1',
        title: '',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Title is required');
    });

    it('requires a parent page ID for createPage', async () => {
      const bubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: '',
        title: 'Hello',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Parent page ID is required');
    });

    it('requires at least one block for appendBlocks', async () => {
      const bubble = new NotionBubble({
        operation: 'appendBlocks',
        blockId: 'page_123',
        blocks: [],
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('At least one block is required');
    });

    it('requires a non-empty search query', async () => {
      const bubble = new NotionBubble({
        operation: 'search',
        query: '',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Search query is required');
    });

    it('rejects an invalid cover URL for createPage', async () => {
      const bubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_1',
        title: 'Hello',
        cover: 'not-a-url',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('cover');
    });

    it('rejects a getBlocks pageSize above 100', async () => {
      const bubble = new NotionBubble({
        operation: 'getBlocks',
        blockId: 'block_123',
        pageSize: 500,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('pageSize');
    });
  });

  // ========================================
  // PAGE OPERATIONS
  // ========================================
  describe('createPage', () => {
    it('creates a page under a parent page', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse(PAGE_RESPONSE)
      );

      const bubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_1',
        title: 'Test Page',
        credentials: mockCredentials,
      });

      const { operation, result } = await bubble.performAction();

      expect(operation).toBe('createPage');
      expect(result.success).toBe(true);
      expect(result.pageId).toBe('page_123');
      expect(result.title).toBe('Test Page');
      expect(result.url).toBe('https://notion.so/page_123');

      const [url, init] = vi.mocked(global.fetch).mock.calls[0] as [
        string,
        RequestInit,
      ];
      expect(url).toBe('https://api.notion.com/v1/pages');
      expect(init.method).toBe('POST');
      const headers = init.headers as Record<string, string>;
      expect(headers.Authorization).toBe('Bearer secret_test_notion_token');
      expect(headers['Notion-Version']).toBe('2022-06-28');

      const body = JSON.parse(init.body as string);
      expect(body.parent).toEqual({ type: 'page_id', page_id: 'parent_1' });
    });

    it('maps an emoji icon and an http icon differently', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce(jsonResponse(PAGE_RESPONSE))
        .mockResolvedValueOnce(jsonResponse(PAGE_RESPONSE));

      const emojiBubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_1',
        title: 'Emoji',
        icon: '🚀',
        credentials: mockCredentials,
      });
      await emojiBubble.performAction();

      expect(
        JSON.parse(vi.mocked(global.fetch).mock.calls[0][1]?.body as string).icon
      ).toEqual({ type: 'emoji', emoji: '🚀' });

      const urlBubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_1',
        title: 'External',
        icon: 'https://example.com/icon.png',
        credentials: mockCredentials,
      });
      await urlBubble.performAction();

      expect(
        JSON.parse(vi.mocked(global.fetch).mock.calls[1][1]?.body as string).icon
      ).toEqual({
        type: 'external',
        external: { url: 'https://example.com/icon.png' },
      });
    });

    it('strips script tags from the title (content sanitization)', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ ...PAGE_RESPONSE, properties: {} })
      );

      const bubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_1',
        title: 'Safe<script>alert("xss")</script>Title',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      const body = JSON.parse(
        vi.mocked(global.fetch).mock.calls[0][1]?.body as string
      );
      const sentTitle = body.properties.title.title[0].text.content;
      expect(sentTitle).toBe('SafeTitle');
      expect(sentTitle).not.toContain('<script>');
    });

    it('returns a controlled failure when the API rejects the create', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(400, 'body.parent.page_id should be a valid uuid')
      );

      const bubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_1',
        title: 'Test Page',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.pageId).toBe('');
      expect(result.error).toContain('Notion API error: 400');
    });
  });

  describe('getPage', () => {
    it('retrieves a page successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse(PAGE_RESPONSE)
      );

      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.pageId).toBe('page_123');
      expect(result.title).toBe('Test Page');
      expect(result.archived).toBe(false);

      expect(vi.mocked(global.fetch).mock.calls[0][0]).toBe(
        'https://api.notion.com/v1/pages/page_123'
      );
    });

    it('returns a controlled failure for a missing page', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(404, 'Could not find page')
      );

      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'nonexistent',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.pageId).toBe('nonexistent');
      expect(result.error).toContain('404');
    });
  });

  describe('updatePage', () => {
    it('updates page properties', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse(PAGE_RESPONSE)
      );

      const bubble = new NotionBubble({
        operation: 'updatePage',
        pageId: 'page_123',
        properties: { Status: { select: { name: 'Done' } } },
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);

      const init = vi.mocked(global.fetch).mock.calls[0][1] as RequestInit;
      expect(init.method).toBe('PATCH');
      expect(JSON.parse(init.body as string).properties).toEqual({
        Status: { select: { name: 'Done' } },
      });
    });
  });

  describe('deletePage', () => {
    it('archives a page by default', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse({}));

      const bubble = new NotionBubble({
        operation: 'deletePage',
        pageId: 'page_123',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.pageId).toBe('page_123');
      expect(result.archived).toBe(true);

      const init = vi.mocked(global.fetch).mock.calls[0][1] as RequestInit;
      expect(init.method).toBe('PATCH');
      expect(JSON.parse(init.body as string)).toEqual({ archived: true });
    });
  });

  // ========================================
  // DATABASE OPERATIONS
  // ========================================
  describe('queryDatabase', () => {
    it('queries a database successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          results: [{ id: 'row_1' }, { id: 'row_2' }],
          next_cursor: 'cursor_2',
          has_more: true,
        })
      );

      const bubble = new NotionBubble({
        operation: 'queryDatabase',
        databaseId: 'db_123',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.results).toHaveLength(2);
      expect(result.totalCount).toBe(2);
      expect(result.hasMore).toBe(true);
      expect(result.nextCursor).toBe('cursor_2');

      expect(vi.mocked(global.fetch).mock.calls[0][0]).toBe(
        'https://api.notion.com/v1/databases/db_123/query'
      );
    });

    it('forwards filter, sorts and startCursor', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ results: [], has_more: false })
      );

      const filter = { property: 'Status', select: { equals: 'Done' } };
      const sorts = [{ property: 'Name', direction: 'ascending' }];

      const bubble = new NotionBubble({
        operation: 'queryDatabase',
        databaseId: 'db_123',
        filter,
        sorts,
        startCursor: 'cursor_1',
        pageSize: 25,
        credentials: mockCredentials,
      });

      await bubble.performAction();

      const body = JSON.parse(
        vi.mocked(global.fetch).mock.calls[0][1]?.body as string
      );
      expect(body.filter).toEqual(filter);
      expect(body.sorts).toEqual(sorts);
      expect(body.start_cursor).toBe('cursor_1');
      expect(body.page_size).toBe(25);
    });

    it('returns an empty result set without error', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ results: [], has_more: false })
      );

      const bubble = new NotionBubble({
        operation: 'queryDatabase',
        databaseId: 'db_123',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.results).toEqual([]);
      expect(result.totalCount).toBe(0);
      expect(result.hasMore).toBe(false);
    });

    it('returns a controlled failure when the query fails', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(404, 'Could not find database')
      );

      const bubble = new NotionBubble({
        operation: 'queryDatabase',
        databaseId: 'nonexistent',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.results).toEqual([]);
      expect(result.error).toContain('404');
    });
  });

  describe('createDatabase', () => {
    it('creates a database with a Name title property', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          id: 'db_new',
          url: 'https://notion.so/db_new',
          properties: { Name: { title: [{ text: { content: 'Tasks' } }] } },
        })
      );

      const bubble = new NotionBubble({
        operation: 'createDatabase',
        parentId: 'parent_1',
        title: 'Tasks',
        properties: { Status: { select: {} } },
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.databaseId).toBe('db_new');
      expect(result.title).toBe('Tasks');

      const body = JSON.parse(
        vi.mocked(global.fetch).mock.calls[0][1]?.body as string
      );
      expect(body.parent).toEqual({ type: 'page_id', page_id: 'parent_1' });
      expect(body.properties.Name.title[0].text.content).toBe('Tasks');
      expect(body.properties.Status).toEqual({ select: {} });
    });
  });

  describe('createDatabaseEntry', () => {
    it('creates a row with a database_id parent', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse(PAGE_RESPONSE)
      );

      const bubble = new NotionBubble({
        operation: 'createDatabaseEntry',
        databaseId: 'db_123',
        properties: { Name: { title: [{ text: { content: 'Row' } }] } },
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.pageId).toBe('page_123');

      const body = JSON.parse(
        vi.mocked(global.fetch).mock.calls[0][1]?.body as string
      );
      expect(body.parent).toEqual({
        type: 'database_id',
        database_id: 'db_123',
      });
    });
  });

  // ========================================
  // BLOCK OPERATIONS
  // ========================================
  describe('appendBlocks', () => {
    it('appends blocks successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ children: [{ id: 'b1' }, { id: 'b2' }] })
      );

      const bubble = new NotionBubble({
        operation: 'appendBlocks',
        blockId: 'page_123',
        blocks: [
          {
            object: 'block',
            type: 'paragraph',
            paragraph: { rich_text: [{ text: { content: 'Hello' } }] },
          },
          {
            object: 'block',
            type: 'paragraph',
            paragraph: { rich_text: [{ text: { content: 'World' } }] },
          },
        ],
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.blockId).toBe('page_123');
      expect(result.appendedBlocks).toBe(2);

      const [url, init] = vi.mocked(global.fetch).mock.calls[0] as [
        string,
        RequestInit,
      ];
      expect(url).toBe(
        'https://api.notion.com/v1/blocks/page_123/children'
      );
      expect(init.method).toBe('PATCH');
    });

    it('sanitizes script/iframe content inside appended blocks', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ children: [] })
      );

      const bubble = new NotionBubble({
        operation: 'appendBlocks',
        blockId: 'page_123',
        blocks: [
          {
            object: 'block',
            type: 'paragraph',
            paragraph: {
              rich_text: [
                {
                  text: {
                    content: 'ok<script>steal()</script><iframe src="x"></iframe>',
                  },
                },
              ],
            },
          },
        ],
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      const sent = vi.mocked(global.fetch).mock.calls[0][1]?.body as string;
      expect(sent).not.toContain('<script>');
      expect(sent).not.toContain('<iframe');
      expect(sent).toContain('ok');
    });
  });

  describe('getBlocks', () => {
    it('lists child blocks with pagination params', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          results: [{ id: 'b1' }],
          next_cursor: 'cursor_2',
          has_more: true,
        })
      );

      const bubble = new NotionBubble({
        operation: 'getBlocks',
        blockId: 'page_123',
        pageSize: 50,
        startCursor: 'cursor_1',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.blocks).toHaveLength(1);
      expect(result.hasMore).toBe(true);
      expect(result.nextCursor).toBe('cursor_2');

      const url = vi.mocked(global.fetch).mock.calls[0][0] as string;
      expect(url).toContain('page_size=50');
      expect(url).toContain('start_cursor=cursor_1');
    });
  });

  describe('getBlock', () => {
    it('retrieves a single block and its typed content', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          id: 'block_123',
          type: 'paragraph',
          paragraph: { rich_text: [{ text: { content: 'Hi' } }] },
          created_time: '2024-01-01T00:00:00.000Z',
          last_edited_time: '2024-01-01T01:00:00.000Z',
          archived: false,
        })
      );

      const bubble = new NotionBubble({
        operation: 'getBlock',
        blockId: 'block_123',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.blockId).toBe('block_123');
      expect(result.type).toBe('paragraph');
      expect(result.content).toEqual({
        rich_text: [{ text: { content: 'Hi' } }],
      });
    });
  });

  describe('updateBlock', () => {
    it('updates a block using its type as the body key', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          id: 'block_123',
          type: 'heading_2',
          heading_2: { rich_text: [{ text: { content: 'Updated' } }] },
          last_edited_time: '2024-01-01T02:00:00.000Z',
          archived: false,
        })
      );

      const bubble = new NotionBubble({
        operation: 'updateBlock',
        blockId: 'block_123',
        type: 'heading_2',
        content: { rich_text: [{ text: { content: 'Updated' } }] },
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.blockId).toBe('block_123');

      const init = vi.mocked(global.fetch).mock.calls[0][1] as RequestInit;
      expect(init.method).toBe('PATCH');
      expect(JSON.parse(init.body as string).heading_2).toEqual({
        rich_text: [{ text: { content: 'Updated' } }],
      });
    });
  });

  // ========================================
  // SEARCH
  // ========================================
  describe('search', () => {
    it('searches successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          results: [
            { object: 'page', id: 'page_1' },
            { object: 'page', id: 'page_2' },
          ],
          next_cursor: null,
          has_more: false,
        })
      );

      const bubble = new NotionBubble({
        operation: 'search',
        query: 'Test',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.results).toHaveLength(2);
      expect(result.totalCount).toBe(2);

      const [url, init] = vi.mocked(global.fetch).mock.calls[0] as [
        string,
        RequestInit,
      ];
      expect(url).toBe('https://api.notion.com/v1/search');
      expect(JSON.parse(init.body as string).query).toBe('Test');
    });

    it('forwards the object-type filter', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ results: [], has_more: false })
      );

      const bubble = new NotionBubble({
        operation: 'search',
        query: 'Tasks',
        filter: { value: 'database', property: 'object' },
        credentials: mockCredentials,
      });

      await bubble.performAction();

      expect(
        JSON.parse(vi.mocked(global.fetch).mock.calls[0][1]?.body as string)
          .filter
      ).toEqual({ value: 'database', property: 'object' });
    });

    it('handles empty search results', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ results: [], has_more: false })
      );

      const bubble = new NotionBubble({
        operation: 'search',
        query: 'nothing-matches',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(true);
      expect(result.results).toEqual([]);
      expect(result.totalCount).toBe(0);
    });
  });

  // ========================================
  // ERROR HANDLING
  // ========================================
  describe('Error Handling', () => {
    it('surfaces an authentication error as a controlled failure', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(401, 'API token is invalid.')
      );

      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('401');
    });

    it('surfaces a network error as a controlled failure', async () => {
      vi.mocked(global.fetch).mockRejectedValueOnce(
        new Error('Network error: ECONNRESET')
      );

      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });

    it('surfaces a rate-limit (429) as RATE_LIMITED after waiting', async () => {
      vi.useFakeTimers();
      try {
        vi.mocked(global.fetch).mockResolvedValueOnce({
          ok: false,
          status: 429,
          statusText: 'Too Many Requests',
          headers: new Headers({ 'Retry-After': '1' }),
          json: async () => ({}),
          text: async () => 'rate limited',
        } as unknown as Response);

        const bubble = new NotionBubble({
          operation: 'getPage',
          pageId: 'page_123',
          credentials: mockCredentials,
        });

        const pending = bubble.performAction();
        await vi.advanceTimersByTimeAsync(2000);
        const { result } = await pending;

        expect(result.success).toBe(false);
        expect(result.error).toBe('RATE_LIMITED');
      } finally {
        vi.useRealTimers();
      }
    });

    it('does not leak the API token in error messages', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(403, 'Insufficient permissions')
      );

      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        credentials: mockCredentials,
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).not.toContain('secret_test_notion_token');
    });

    it('reports a missing NOTION_OAUTH_TOKEN entry', async () => {
      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        // credentials object present but missing the Notion entry
        credentials: { [CredentialType.OPENAI_CRED]: 'sk-test' },
      });

      const { result } = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Notion API key is required');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('throws from chooseCredential when the credentials map is absent', async () => {
      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
      });

      // chooseCredential() throws, and performAction() calls it before its
      // try/catch, so the error propagates.
      await expect(bubble.performAction()).rejects.toThrow(
        'Notion API credentials are required'
      );
    });
  });

  // ========================================
  // CREDENTIALS
  // ========================================
  describe('testCredential', () => {
    it('resolves true when users/me succeeds', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ object: 'user', id: 'bot_1' })
      );

      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        credentials: mockCredentials,
      });

      await expect(bubble.testCredential()).resolves.toBe(true);
      expect(vi.mocked(global.fetch).mock.calls[0][0]).toBe(
        'https://api.notion.com/v1/users/me'
      );
    });

    it('resolves false when users/me fails', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(401, 'API token is invalid.')
      );

      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        credentials: mockCredentials,
      });

      await expect(bubble.testCredential()).resolves.toBe(false);
    });

    it('resolves false when the Notion token entry is absent', async () => {
      const bubble = new NotionBubble({
        operation: 'getPage',
        pageId: 'page_123',
        credentials: { [CredentialType.OPENAI_CRED]: 'sk-test' },
      });

      await expect(bubble.testCredential()).resolves.toBe(false);
      expect(global.fetch).not.toHaveBeenCalled();
    });
  });
});
