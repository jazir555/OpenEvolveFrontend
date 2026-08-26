/**
 * Comprehensive tests for the Notion Bubble (src/bubbles/service-bubble/notion/notion.ts)
 *
 * Tests all major operations:
 * - Page operations (retrieve, create, update, archive)
 * - Database / data-source operations (retrieve, query)
 * - Block operations (append, retrieve children, retrieve, update/archive)
 * - User operations (list)
 * - Search operations
 *
 * CONTRACT NOTES (the previous version of this file was written against a
 * contract this bubble never had, so 37/39 tests failed):
 *  - `performAction()` returns a FLAT result: `{ operation, success, error, ... }`.
 *    There is no `result.result` wrapper (that is the google-sheets bubble's shape).
 *  - Operation names are snake_case (`retrieve_page`, not `retrievePage`) and IDs
 *    are snake_case params (`page_id`, `block_id`, `data_source_id`, ...).
 *  - Notion has no `delete_page` / `delete_block` / `retrieve_user` operations;
 *    archiving is done via `update_page` / `update_block` with `archived: true`.
 *  - Credentials are `NOTION_OAUTH_TOKEN` or `NOTION_API` (there is no
 *    `NOTION_CRED`).
 *  - The bubble does NOT retry: `makeNotionApiCall` issues exactly one fetch.
 *  - The Bubble base constructor does not throw on invalid params; it records a
 *    validation error that `action()` reports as `{ success: false, error }`.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { NotionBubble } from './notion/notion.js';
import { CredentialType } from '@bubblelab/shared-schemas';

const mockCredentials = {
  [CredentialType.NOTION_API]: 'secret_test_api_key',
};

function jsonResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => body,
  } as unknown as Response;
}

function errorResponse(status: number, body: unknown) {
  return {
    ok: false,
    status,
    statusText: 'Error',
    json: async () => body,
  } as unknown as Response;
}

/** Minimal Notion user object (satisfies UserSchema). */
const mockUser = {
  object: 'user' as const,
  id: 'user_123',
  type: 'person' as const,
  name: 'John Doe',
};

/** Minimal Notion page object. */
function mockPage(overrides: Record<string, unknown> = {}) {
  return {
    object: 'page',
    id: 'page_test_123',
    created_time: '2024-01-01T00:00:00.000Z',
    last_edited_time: '2024-01-01T01:00:00.000Z',
    created_by: mockUser,
    last_edited_by: mockUser,
    parent: { type: 'workspace', workspace: true },
    archived: false,
    properties: {
      title: { title: [{ text: { content: 'Test Page' } }] },
    },
    url: 'https://notion.so/page_test_123',
    ...overrides,
  };
}

/** Minimal Notion block object. */
function mockBlock(overrides: Record<string, unknown> = {}) {
  return {
    object: 'block',
    id: 'block_123',
    created_time: '2024-01-01T00:00:00.000Z',
    last_edited_time: '2024-01-01T01:00:00.000Z',
    created_by: mockUser,
    last_edited_by: mockUser,
    has_children: false,
    archived: false,
    type: 'paragraph',
    ...overrides,
  };
}

describe('NotionBubble', () => {
  let notionBubble: NotionBubble;

  beforeEach(() => {
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
      expect(NotionBubble.alias).toBe('notion');
      expect(NotionBubble.schema).toBeDefined();
      expect(NotionBubble.resultSchema).toBeDefined();
    });

    it('defaults to list_users when constructed with no params', () => {
      const bubble = new NotionBubble();
      expect(bubble.currentParams.operation).toBe('list_users');
    });

    it('excludes credentials from currentParams', () => {
      const bubble = new NotionBubble({
        operation: 'list_users',
        credentials: mockCredentials,
      });
      expect(
        (bubble.currentParams as Record<string, unknown>).credentials
      ).toBeUndefined();
    });
  });

  // ========================================
  // PAGE OPERATIONS
  // ========================================
  describe('Page Operations', () => {
    describe('retrieve_page', () => {
      it('should retrieve a page successfully', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(mockPage()));

        notionBubble = new NotionBubble({
          operation: 'retrieve_page',
          page_id: 'page_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.error).toBe('');
        expect(result.page).toBeDefined();
        expect(result.page!.id).toBe('page_test_123');
      });

      it('should send the bearer token and Notion-Version header', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(mockPage()));

        notionBubble = new NotionBubble({
          operation: 'retrieve_page',
          page_id: 'page_test_123',
          credentials: mockCredentials,
        });

        await notionBubble.performAction();

        const [url, init] = vi.mocked(fetch).mock.calls[0] as [
          string,
          RequestInit,
        ];
        expect(url).toBe('https://api.notion.com/v1/pages/page_test_123');
        expect(init.method).toBe('GET');
        const headers = init.headers as Record<string, string>;
        expect(headers.Authorization).toBe('Bearer secret_test_api_key');
        expect(headers['Notion-Version']).toBeDefined();
      });

      it('should forward filter_properties as query params', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(mockPage()));

        notionBubble = new NotionBubble({
          operation: 'retrieve_page',
          page_id: 'page_test_123',
          filter_properties: ['abc', 'def'],
          credentials: mockCredentials,
        });

        await notionBubble.performAction();

        const url = vi.mocked(fetch).mock.calls[0][0] as string;
        expect(url).toContain('filter_properties=abc');
        expect(url).toContain('filter_properties=def');
      });

      it('should handle a non-existent page', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          errorResponse(404, { message: 'Could not find page' })
        );

        notionBubble = new NotionBubble({
          operation: 'retrieve_page',
          page_id: 'nonexistent',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.error).toContain('Could not find page');
      });

      it('should report a missing page_id as a controlled validation error', async () => {
        const bubble = new NotionBubble({
          // @ts-expect-error page_id is required for retrieve_page
          operation: 'retrieve_page',
          credentials: mockCredentials,
        });

        const result = await bubble.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('page_id');
        expect(global.fetch).not.toHaveBeenCalled();
      });
    });

    describe('create_page', () => {
      it('should create a page under a workspace/page parent', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse(mockPage({ id: 'page_new_456' }))
        );

        notionBubble = new NotionBubble({
          operation: 'create_page',
          parent: { type: 'page_id', page_id: 'parent_page_1' },
          properties: {
            title: { title: [{ text: { content: 'New Page' } }] },
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.page!.id).toBe('page_new_456');

        const [url, init] = vi.mocked(fetch).mock.calls[0] as [
          string,
          RequestInit,
        ];
        expect(url).toBe('https://api.notion.com/v1/pages');
        expect(init.method).toBe('POST');
        expect(JSON.parse(init.body as string).parent).toEqual({
          type: 'page_id',
          page_id: 'parent_page_1',
        });
      });

      it('should auto-resolve a database_id parent to its data_source_id', async () => {
        vi.mocked(fetch)
          // 1) retrieve_database to resolve the data source
          .mockResolvedValueOnce(
            jsonResponse({
              object: 'database',
              id: 'db_1',
              data_sources: [{ id: 'ds_1', name: 'Tasks' }],
            })
          )
          // 2) the actual page creation
          .mockResolvedValueOnce(
            jsonResponse(mockPage({ id: 'page_in_db' }))
          );

        notionBubble = new NotionBubble({
          operation: 'create_page',
          parent: { type: 'database_id', database_id: 'db_1' },
          properties: { Name: { title: [{ text: { content: 'Row' } }] } },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.page!.id).toBe('page_in_db');
        expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2);

        expect(vi.mocked(fetch).mock.calls[0][0]).toBe(
          'https://api.notion.com/v1/databases/db_1'
        );
        const body = JSON.parse(
          vi.mocked(fetch).mock.calls[1][1]?.body as string
        );
        expect(body.parent).toEqual({
          type: 'data_source_id',
          data_source_id: 'ds_1',
        });
      });

      it('should report a missing parent as a controlled validation error', async () => {
        const bubble = new NotionBubble({
          // @ts-expect-error parent is required for create_page
          operation: 'create_page',
          properties: {},
          credentials: mockCredentials,
        });

        const result = await bubble.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('parent');
        expect(global.fetch).not.toHaveBeenCalled();
      });

      it('should surface an API validation error from Notion', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          errorResponse(400, {
            message: 'body.properties.Name should be defined',
          })
        );

        notionBubble = new NotionBubble({
          operation: 'create_page',
          parent: { type: 'data_source_id', data_source_id: 'ds_1' },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.error).toContain('should be defined');
      });
    });

    describe('update_page', () => {
      it('should update page properties successfully', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse(
            mockPage({
              properties: {
                title: { title: [{ text: { content: 'Renamed' } }] },
              },
            })
          )
        );

        notionBubble = new NotionBubble({
          operation: 'update_page',
          page_id: 'page_test_123',
          properties: {
            title: { title: [{ text: { content: 'Renamed' } }] },
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.page!.id).toBe('page_test_123');

        const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit;
        expect(init.method).toBe('PATCH');
        expect(JSON.parse(init.body as string).properties).toBeDefined();
      });

      it('should archive a page via archived: true', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse(mockPage({ archived: true }))
        );

        notionBubble = new NotionBubble({
          operation: 'update_page',
          page_id: 'page_test_123',
          archived: true,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.page!.archived).toBe(true);
        expect(
          JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string).archived
        ).toBe(true);
      });

      it('should handle page not found on update', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          errorResponse(404, { message: 'Could not find page with ID' })
        );

        notionBubble = new NotionBubble({
          operation: 'update_page',
          page_id: 'nonexistent',
          archived: true,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.error).toContain('Could not find page');
      });
    });
  });

  // ========================================
  // DATABASE / DATA SOURCE OPERATIONS
  // ========================================
  describe('Database Operations', () => {
    describe('retrieve_database', () => {
      it('should retrieve a database and merge in the data-source schema', async () => {
        vi.mocked(fetch)
          .mockResolvedValueOnce(
            jsonResponse({
              object: 'database',
              id: 'db_test_123',
              created_time: '2024-01-01T00:00:00.000Z',
              last_edited_time: '2024-01-01T01:00:00.000Z',
              title: [{ plain_text: 'Test Database' }],
              parent: { type: 'workspace', workspace: true },
              data_sources: [{ id: 'ds_1', name: 'Tasks' }],
            })
          )
          .mockResolvedValueOnce(
            jsonResponse({
              properties: { Name: { type: 'title' }, Done: { type: 'checkbox' } },
            })
          );

        notionBubble = new NotionBubble({
          operation: 'retrieve_database',
          database_id: 'db_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.database!.id).toBe('db_test_123');
        expect(
          (result.database as unknown as Record<string, unknown>).properties
        ).toEqual({ Name: { type: 'title' }, Done: { type: 'checkbox' } });
        expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2);
      });

      it('should still succeed when the schema fetch fails', async () => {
        vi.mocked(fetch)
          .mockResolvedValueOnce(
            jsonResponse({
              object: 'database',
              id: 'db_test_123',
              data_sources: [{ id: 'ds_1', name: 'Tasks' }],
            })
          )
          .mockResolvedValueOnce(
            errorResponse(403, { message: 'No access to data source' })
          );

        notionBubble = new NotionBubble({
          operation: 'retrieve_database',
          database_id: 'db_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.database!.id).toBe('db_test_123');
      });

      it('should handle a non-existent database', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          errorResponse(404, { message: 'Could not find database' })
        );

        notionBubble = new NotionBubble({
          operation: 'retrieve_database',
          database_id: 'nonexistent',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.error).toContain('Could not find database');
      });
    });

    describe('query_data_source', () => {
      it('should query a data source successfully', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [
              { object: 'page', id: 'page_1' },
              { object: 'page', id: 'page_2' },
            ],
            next_cursor: null,
            has_more: false,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'query_data_source',
          data_source_id: 'ds_1',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.results).toHaveLength(2);
        expect(result.has_more).toBe(false);

        expect(vi.mocked(fetch).mock.calls[0][0]).toContain(
          'data_sources/ds_1/query'
        );
      });

      it('should forward filter and sorts to the API', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [],
            next_cursor: null,
            has_more: false,
          })
        );

        const filter = { property: 'Status', select: { equals: 'Done' } };
        const sorts = [{ property: 'Name', direction: 'ascending' }];

        notionBubble = new NotionBubble({
          operation: 'query_data_source',
          data_source_id: 'ds_1',
          filter,
          sorts,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        const body = JSON.parse(
          vi.mocked(fetch).mock.calls[0][1]?.body as string
        );
        expect(body.filter).toEqual(filter);
        expect(body.sorts).toEqual(sorts);
      });

      it('should handle pagination via start_cursor', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [{ object: 'page', id: 'page_3' }],
            next_cursor: 'cursor_2',
            has_more: true,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'query_data_source',
          data_source_id: 'ds_1',
          start_cursor: 'cursor_1',
          page_size: 1,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.has_more).toBe(true);
        expect(result.next_cursor).toBe('cursor_2');

        const body = JSON.parse(
          vi.mocked(fetch).mock.calls[0][1]?.body as string
        );
        expect(body.start_cursor).toBe('cursor_1');
        expect(body.page_size).toBe(1);
      });

      it('should resolve a database_id to a data_source_id when needed', async () => {
        vi.mocked(fetch)
          .mockResolvedValueOnce(
            jsonResponse({
              object: 'database',
              id: 'db_1',
              data_sources: [{ id: 'ds_resolved', name: 'Tasks' }],
            })
          )
          .mockResolvedValueOnce(
            jsonResponse({
              object: 'list',
              results: [],
              next_cursor: null,
              has_more: false,
            })
          );

        notionBubble = new NotionBubble({
          operation: 'query_data_source',
          database_id: 'db_1',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(vi.mocked(fetch).mock.calls[1][0]).toContain(
          'data_sources/ds_resolved/query'
        );
      });
    });
  });

  // ========================================
  // BLOCK OPERATIONS
  // ========================================
  describe('Block Operations', () => {
    describe('append_block_children', () => {
      it('should append blocks successfully', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [mockBlock()],
            next_cursor: null,
            has_more: false,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'append_block_children',
          block_id: 'page_test_123',
          children: [
            {
              object: 'block',
              type: 'paragraph',
              paragraph: {
                rich_text: [{ type: 'text', text: { content: 'Hello' } }],
              },
            },
          ],
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.blocks).toHaveLength(1);

        const [url, init] = vi.mocked(fetch).mock.calls[0] as [
          string,
          RequestInit,
        ];
        expect(url).toBe(
          'https://api.notion.com/v1/blocks/page_test_123/children'
        );
        expect(init.method).toBe('PATCH');
      });

      it('should append multiple blocks', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [
              mockBlock({ id: 'b1' }),
              mockBlock({ id: 'b2' }),
              mockBlock({ id: 'b3' }),
            ],
            next_cursor: null,
            has_more: false,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'append_block_children',
          block_id: 'page_test_123',
          children: [
            { object: 'block', type: 'paragraph', paragraph: {} },
            { object: 'block', type: 'paragraph', paragraph: {} },
            { object: 'block', type: 'paragraph', paragraph: {} },
          ],
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.blocks).toHaveLength(3);
      });

      it('should reject an empty children array', async () => {
        const bubble = new NotionBubble({
          operation: 'append_block_children',
          block_id: 'page_test_123',
          children: [],
          credentials: mockCredentials,
        });

        const result = await bubble.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('children');
        expect(global.fetch).not.toHaveBeenCalled();
      });

      it('should reject more than 100 children', async () => {
        const bubble = new NotionBubble({
          operation: 'append_block_children',
          block_id: 'page_test_123',
          children: Array.from({ length: 101 }, () => ({
            object: 'block',
            type: 'paragraph',
          })),
          credentials: mockCredentials,
        });

        const result = await bubble.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('children');
      });
    });

    describe('retrieve_block_children', () => {
      it('should retrieve block children successfully', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [mockBlock({ id: 'b1' }), mockBlock({ id: 'b2' })],
            next_cursor: 'cursor_next',
            has_more: true,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'retrieve_block_children',
          block_id: 'page_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.blocks).toHaveLength(2);
        expect(result.has_more).toBe(true);
        expect(result.next_cursor).toBe('cursor_next');
      });

      it('should handle pagination with a start cursor', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [mockBlock({ id: 'b3' })],
            next_cursor: null,
            has_more: false,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'retrieve_block_children',
          block_id: 'page_test_123',
          start_cursor: 'cursor_next',
          page_size: 50,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);

        const url = vi.mocked(fetch).mock.calls[0][0] as string;
        expect(url).toContain('start_cursor=cursor_next');
        expect(url).toContain('page_size=50');
      });
    });

    describe('retrieve_block', () => {
      it('should retrieve a single block', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(mockBlock()));

        notionBubble = new NotionBubble({
          operation: 'retrieve_block',
          block_id: 'block_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.block!.id).toBe('block_123');
      });
    });

    describe('update_block', () => {
      it('should archive a block via archived: true', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse(mockBlock({ archived: true }))
        );

        notionBubble = new NotionBubble({
          operation: 'update_block',
          block_id: 'block_123',
          archived: true,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.block!.archived).toBe(true);

        const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit;
        expect(init.method).toBe('PATCH');
        expect(JSON.parse(init.body as string)).toEqual({ archived: true });
      });
    });
  });

  // ========================================
  // USER OPERATIONS
  // ========================================
  describe('User Operations', () => {
    describe('list_users', () => {
      it('should list users successfully', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [
              mockUser,
              { object: 'user', id: 'user_456', type: 'bot', name: 'Bot' },
            ],
            next_cursor: 'cursor_users',
            has_more: true,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'list_users',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.users).toHaveLength(2);
        expect(result.users![0].name).toBe('John Doe');
        expect(result.has_more).toBe(true);
      });

      it('should handle pagination', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [mockUser],
            next_cursor: null,
            has_more: false,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'list_users',
          start_cursor: 'cursor_users',
          page_size: 10,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);

        const url = vi.mocked(fetch).mock.calls[0][0] as string;
        expect(url).toContain('start_cursor=cursor_users');
        expect(url).toContain('page_size=10');
      });

      it('should surface an unauthorized error', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          errorResponse(401, { message: 'API token is invalid.' })
        );

        notionBubble = new NotionBubble({
          operation: 'list_users',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.error).toContain('API token is invalid');
      });
    });
  });

  // ========================================
  // SEARCH OPERATIONS
  // ========================================
  describe('Search Operations', () => {
    describe('search', () => {
      it('should search pages successfully', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [
              {
                object: 'page',
                id: 'page_1',
                created_time: '2024-01-01T00:00:00.000Z',
                last_edited_time: '2024-01-01T01:00:00.000Z',
              },
              {
                object: 'page',
                id: 'page_2',
                created_time: '2024-01-01T00:00:00.000Z',
                last_edited_time: '2024-01-01T01:00:00.000Z',
              },
            ],
            next_cursor: null,
            has_more: false,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'search',
          query: 'Test',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.results).toHaveLength(2);

        const [url, init] = vi.mocked(fetch).mock.calls[0] as [
          string,
          RequestInit,
        ];
        expect(url).toBe('https://api.notion.com/v1/search');
        expect(init.method).toBe('POST');
        expect(JSON.parse(init.body as string).query).toBe('Test');
      });

      it('should search data sources via the object filter', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [
              {
                object: 'data_source',
                id: 'ds_1',
                created_time: '2024-01-01T00:00:00.000Z',
                last_edited_time: '2024-01-01T01:00:00.000Z',
              },
            ],
            next_cursor: null,
            has_more: false,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'search',
          query: 'Tasks',
          filter: { value: 'data_source', property: 'object' },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.results).toHaveLength(1);
        expect(result.results![0].object).toBe('data_source');
        expect(
          JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string).filter
        ).toEqual({ value: 'data_source', property: 'object' });
      });

      it('should handle empty search results', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [],
            next_cursor: null,
            has_more: false,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'search',
          query: 'nothing-matches-this',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.results).toHaveLength(0);
      });

      it('should handle pagination in search', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          jsonResponse({
            object: 'list',
            results: [
              {
                object: 'page',
                id: 'page_9',
                created_time: '2024-01-01T00:00:00.000Z',
                last_edited_time: '2024-01-01T01:00:00.000Z',
              },
            ],
            next_cursor: 'cursor_search',
            has_more: true,
          })
        );

        notionBubble = new NotionBubble({
          operation: 'search',
          query: 'Test',
          start_cursor: 'cursor_prev',
          page_size: 1,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.has_more).toBe(true);
        expect(result.next_cursor).toBe('cursor_search');

        const body = JSON.parse(
          vi.mocked(fetch).mock.calls[0][1]?.body as string
        );
        expect(body.start_cursor).toBe('cursor_prev');
        expect(body.page_size).toBe(1);
      });

      it('should reject an invalid filter value', async () => {
        const bubble = new NotionBubble({
          operation: 'search',
          // @ts-expect-error 'database' is not an allowed filter value
          filter: { value: 'database', property: 'object' },
          credentials: mockCredentials,
        });

        const result = await bubble.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('filter');
      });
    });
  });

  // ========================================
  // ERROR HANDLING
  // ========================================
  describe('Error Handling', () => {
    it('should handle authentication errors', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        errorResponse(401, { message: 'Unauthorized' })
      );

      notionBubble = new NotionBubble({
        operation: 'retrieve_page',
        page_id: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unauthorized');
    });

    it('should surface rate limiting (429) without retrying', async () => {
      // makeNotionApiCall performs exactly one fetch — there is no retry loop.
      vi.mocked(fetch).mockResolvedValueOnce(
        errorResponse(429, { message: 'Rate limited' })
      );

      notionBubble = new NotionBubble({
        operation: 'retrieve_page',
        page_id: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Rate limited');
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1);
    });

    it('should handle network errors', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));

      notionBubble = new NotionBubble({
        operation: 'retrieve_page',
        page_id: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });

    it('should handle malformed responses', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => {
          throw new SyntaxError('Unexpected token < in JSON');
        },
      } as unknown as Response);

      notionBubble = new NotionBubble({
        operation: 'retrieve_page',
        page_id: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('JSON');
    });

    it('should report a missing token without hitting the network', async () => {
      notionBubble = new NotionBubble({
        operation: 'retrieve_page',
        page_id: 'page_test_123',
      });

      const result = await notionBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Notion OAuth token is required');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('should not leak the API token in error messages', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        errorResponse(403, { message: 'Insufficient permissions' })
      );

      notionBubble = new NotionBubble({
        operation: 'retrieve_page',
        page_id: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).not.toContain('secret_test_api_key');
    });

    it('should report an unknown operation as a controlled validation error', async () => {
      const bubble = new NotionBubble({
        // @ts-expect-error deliberately invalid operation
        operation: 'delete_everything',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  // ========================================
  // CREDENTIAL TESTING
  // ========================================
  describe('Credential Testing', () => {
    it('should test credentials successfully', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        jsonResponse({
          object: 'list',
          results: [mockUser],
          next_cursor: null,
          has_more: false,
        })
      );

      notionBubble = new NotionBubble({
        operation: 'list_users',
        credentials: mockCredentials,
      });

      await expect(notionBubble.testCredential()).resolves.toBe(true);
      expect(vi.mocked(fetch).mock.calls[0][0]).toBe(
        'https://api.notion.com/v1/users'
      );
    });

    it('should propagate the vendor message for invalid credentials', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        errorResponse(401, { message: 'API token is invalid.' })
      );

      notionBubble = new NotionBubble({
        operation: 'list_users',
        credentials: { [CredentialType.NOTION_API]: 'invalid' },
      });

      await expect(notionBubble.testCredential()).rejects.toThrow(
        'API token is invalid.'
      );
    });

    it('should handle missing credentials', async () => {
      notionBubble = new NotionBubble({ operation: 'list_users' });

      await expect(notionBubble.testCredential()).resolves.toBe(false);
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('should accept an OAuth token credential', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        jsonResponse({
          object: 'list',
          results: [],
          next_cursor: null,
          has_more: false,
        })
      );

      notionBubble = new NotionBubble({
        operation: 'list_users',
        credentials: {
          [CredentialType.NOTION_OAUTH_TOKEN]: 'oauth_token_value',
        },
      });

      await expect(notionBubble.testCredential()).resolves.toBe(true);

      const headers = vi.mocked(fetch).mock.calls[0][1]
        ?.headers as Record<string, string>;
      expect(headers.Authorization).toBe('Bearer oauth_token_value');
    });
  });

  // ========================================
  // CONCURRENCY
  // ========================================
  describe('Concurrency', () => {
    it('handles concurrent operations independently', async () => {
      vi.mocked(fetch).mockImplementation(async () =>
        jsonResponse(mockPage())
      );

      const bubbles = Array.from(
        { length: 5 },
        (_, i) =>
          new NotionBubble({
            operation: 'retrieve_page',
            page_id: `page_${i}`,
            credentials: mockCredentials,
          })
      );

      const results = await Promise.all(bubbles.map((b) => b.performAction()));

      expect(results).toHaveLength(5);
      expect(results.every((r) => r.success)).toBe(true);
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(5);
    });
  });
});
