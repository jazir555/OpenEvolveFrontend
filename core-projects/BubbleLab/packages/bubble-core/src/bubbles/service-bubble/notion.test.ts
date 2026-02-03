/**
 * Comprehensive tests for Notion Bubble
 *
 * Tests all major operations:
 * - Page operations (retrieve, create, update, delete)
 * - Database operations (query, retrieve)
 * - Block operations (append, update, delete)
 * - User operations (list, retrieve)
 * - Search operations
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { NotionBubble } from './notion/notion.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('NotionBubble', () => {
  let notionBubble: NotionBubble;
  const mockCredentials = {
    [CredentialType.NOTION_CRED]: 'secret_test_api_key',
  };

  beforeEach(() => {
    // Mock fetch globally
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Page Operations', () => {
    describe('retrievePage', () => {
      it('should retrieve a page successfully', async () => {
        const mockResponse = {
          id: 'page_test_123',
          created_time: '2024-01-01T00:00:00.000Z',
          last_edited_time: '2024-01-01T01:00:00.000Z',
          archived: false,
          properties: {
            title: {
              title: [
                {
                  text: {
                    content: 'Test Page',
                  },
                },
              ],
            },
          },
          object: 'page',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'retrieve_page',
          pageId: 'page_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.page).toBeDefined();
        expect(result.result.page.id).toBe('page_test_123');
      });

      it('should handle non-existent page', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: false,
          status: 404,
          json: async () => ({
            message: 'Not found',
          }),
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'retrieve_page',
          pageId: 'page_nonexistent',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(false);
        expect(result.result.error).toContain('Not found');
      });

      it('should validate page ID format', async () => {
        notionBubble = new NotionBubble({
          operation: 'retrieve_page',
          pageId: 'invalid-id',
          credentials: mockCredentials,
        });

        await expect(notionBubble.performAction()).rejects.toThrow();
      });
    });

    describe('createPage', () => {
      it('should create a page successfully', async () => {
        const mockResponse = {
          id: 'page_new_123',
          created_time: '2024-01-01T00:00:00.000Z',
          last_edited_time: '2024-01-01T00:00:00.000Z',
          archived: false,
          properties: {
            Name: {
              title: [
                {
                  text: {
                    content: 'New Page',
                  },
                },
              ],
            },
          },
          object: 'page',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'create_page',
          parentId: 'parent_test_123',
          parentType: 'page_id',
          properties: {
            Name: {
              title: [
                {
                  text: {
                    content: 'New Page',
                  },
                },
              ],
            },
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.page.id).toBe('page_new_123');
      });

      it('should handle missing required properties', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: false,
          status: 400,
          json: async () => ({
            message: 'validation error',
          }),
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'create_page',
          parentId: 'parent_test_123',
          parentType: 'page_id',
          properties: {},
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(false);
      });

      it('should create page in database', async () => {
        const mockResponse = {
          id: 'page_db_123',
          created_time: '2024-01-01T00:00:00.000Z',
          archived: false,
          properties: {
            'Task Name': {
              title: [
                {
                  text: {
                    content: 'Complete task',
                  },
                },
              ],
            },
            Status: {
              select: {
                name: 'In Progress',
              },
            },
          },
          object: 'page',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'create_page',
          parentId: 'database_test_123',
          parentType: 'database_id',
          properties: {
            'Task Name': {
              title: [
                {
                  text: {
                    content: 'Complete task',
                  },
                },
              ],
            },
            Status: {
              select: {
                name: 'In Progress',
              },
            },
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.page.properties.Status.select.name).toBe('In Progress');
      });
    });

    describe('updatePage', () => {
      it('should update a page successfully', async () => {
        const mockResponse = {
          id: 'page_test_123',
          updated: {
            properties: {
              title: {
                title: [
                  {
                    text: {
                      content: 'Updated Title',
                    },
                  },
                ],
              },
            },
          },
          archived: false,
          object: 'page',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'update_page',
          pageId: 'page_test_123',
          properties: {
            title: {
              title: [
                {
                  text: {
                    content: 'Updated Title',
                  },
                },
              ],
            },
          },
          archived: false,
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
      });

      it('should handle page not found on update', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: false,
          status: 404,
          json: async () => ({
            message: 'Not found',
          }),
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'update_page',
          pageId: 'page_nonexistent',
          properties: {
            title: {
              title: [
                {
                  text: {
                    content: 'Updated',
                  },
                },
              ],
            },
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(false);
      });
    });

    describe('deletePage', () => {
      it('should archive a page successfully', async () => {
        const mockResponse = {
          id: 'page_test_123',
          archived: true,
          object: 'page',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'delete_page',
          pageId: 'page_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.archived).toBe(true);
      });

      it('should handle already archived page', async () => {
        const mockResponse = {
          id: 'page_test_123',
          archived: true,
          object: 'page',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'delete_page',
          pageId: 'page_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
      });
    });
  });

  describe('Database Operations', () => {
    describe('queryDatabase', () => {
      it('should query database successfully', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'page_1',
              properties: {
                Name: {
                  title: [
                    {
                      text: {
                        content: 'Task 1',
                      },
                    },
                  ],
                },
              },
            },
            {
              id: 'page_2',
              properties: {
                Name: {
                  title: [
                    {
                      text: {
                        content: 'Task 2',
                      },
                    },
                  ],
                },
              },
            },
          ],
          next_cursor: 'next_cursor_token',
          has_more: true,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'query_data_source',
          databaseId: 'database_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(2);
        expect(result.result.hasMore).toBe(true);
        expect(result.result.nextCursor).toBe('next_cursor_token');
      });

      it('should query with filter', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'page_1',
              properties: {
                Status: {
                  select: {
                    name: 'In Progress',
                  },
                },
              },
            },
          ],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'query_data_source',
          databaseId: 'database_test_123',
          filter: {
            property: 'Status',
            select: {
              equals: 'In Progress',
            },
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(1);
      });

      it('should query with sorting', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'page_1',
              properties: {
                'Due Date': {
                  date: {
                    start: '2024-01-01',
                  },
                },
              },
            },
          ],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'query_data_source',
          databaseId: 'database_test_123',
          sorts: [
            {
              property: 'Due Date',
              direction: 'ascending',
            },
          ],
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(vi.mocked(fetch)).toHaveBeenCalledWith(
          expect.any(String),
          expect.objectContaining({
            method: 'POST',
          })
        );
      });

      it('should handle pagination', async () => {
        const mockResponse = {
          object: 'list',
          results: [],
          next_cursor: null,
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'query_data_source',
          databaseId: 'database_test_123',
          startCursor: 'start_cursor_token',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.hasMore).toBe(false);
      });
    });

    describe('retrieveDatabase', () => {
      it('should retrieve database successfully', async () => {
        const mockResponse = {
          id: 'database_test_123',
          created_time: '2024-01-01T00:00:00.000Z',
          last_edited_time: '2024-01-01T01:00:00.000Z',
          title: [
            {
              type: 'text',
              text: {
                content: 'Tasks Database',
              },
            },
          ],
          properties: {
            Name: {
              title: {},
            },
            Status: {
              select: {
                options: [
                  { name: 'Not Started', color: 'gray' },
                  { name: 'In Progress', color: 'blue' },
                  { name: 'Completed', color: 'green' },
                ],
              },
            },
          },
          object: 'database',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'retrieve_database',
          databaseId: 'database_test_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.database.id).toBe('database_test_123');
        expect(result.result.database.properties.Status).toBeDefined();
      });

      it('should handle non-existent database', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: false,
          status: 404,
          json: async () => ({
            message: 'Not found',
          }),
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'retrieve_database',
          databaseId: 'database_nonexistent',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(false);
      });
    });
  });

  describe('Block Operations', () => {
    describe('appendBlockChildren', () => {
      it('should append blocks successfully', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'block_1',
              type: 'paragraph',
              paragraph: {
                rich_text: [
                  {
                    type: 'text',
                    text: {
                      content: 'New paragraph',
                    },
                  },
                ],
              },
            },
          ],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'append_block_children',
          blockId: 'block_parent_123',
          children: [
            {
              object: 'block',
              type: 'paragraph',
              paragraph: {
                rich_text: [
                  {
                    type: 'text',
                    text: {
                      content: 'New paragraph',
                    },
                  },
                ],
              },
            },
          ],
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(1);
      });

      it('should append multiple blocks', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'block_1',
              type: 'paragraph',
            },
            {
              id: 'block_2',
              type: 'heading_1',
            },
            {
              id: 'block_3',
              type: 'to_do',
            },
          ],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'append_block_children',
          blockId: 'block_parent_123',
          children: [
            {
              object: 'block',
              type: 'paragraph',
              paragraph: {
                rich_text: [{ type: 'text', text: { content: 'Paragraph' } }],
              },
            },
            {
              object: 'block',
              type: 'heading_1',
              heading_1: {
                rich_text: [{ type: 'text', text: { content: 'Heading' } }],
              },
            },
            {
              object: 'block',
              type: 'to_do',
              to_do: {
                rich_text: [{ type: 'text', text: { content: 'Task' } }],
                checked: false,
              },
            },
          ],
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(3);
      });
    });

    describe('retrieveBlockChildren', () => {
      it('should retrieve block children successfully', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'block_1',
              type: 'paragraph',
              has_children: false,
            },
            {
              id: 'block_2',
              type: 'bulleted_list_item',
              has_children: false,
            },
          ],
          next_cursor: 'next_cursor',
          has_more: true,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'retrieve_block_children',
          blockId: 'block_parent_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(2);
        expect(result.result.hasMore).toBe(true);
      });

      it('should handle pagination with start cursor', async () => {
        const mockResponse = {
          object: 'list',
          results: [],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'retrieve_block_children',
          blockId: 'block_parent_123',
          startCursor: 'cursor_token',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
      });
    });

    describe('updateBlock', () => {
      it('should update block successfully', async () => {
        const mockResponse = {
          id: 'block_123',
          type: 'paragraph',
          paragraph: {
            rich_text: [
              {
                type: 'text',
                text: {
                  content: 'Updated paragraph',
                },
              },
            ],
          },
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'update_block',
          blockId: 'block_123',
          paragraph: {
            rich_text: [
              {
                type: 'text',
                text: {
                  content: 'Updated paragraph',
                },
              },
            ],
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
      });
    });

    describe('deleteBlock', () => {
      it('should delete block successfully (archive)', async () => {
        const mockResponse = {
          id: 'block_123',
          archived: true,
          object: 'block',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'delete_block',
          blockId: 'block_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.archived).toBe(true);
      });
    });
  });

  describe('User Operations', () => {
    describe('listUsers', () => {
      it('should list users successfully', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'user_1',
              name: 'John Doe',
              avatar_url: 'https://example.com/avatar1.jpg',
              type: 'person',
              person: {
                email: 'john@example.com',
              },
            },
            {
              id: 'user_2',
              name: 'Jane Smith',
              avatar_url: 'https://example.com/avatar2.jpg',
              type: 'person',
              person: {
                email: 'jane@example.com',
              },
            },
          ],
          next_cursor: 'next_cursor',
          has_more: true,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'list_users',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(2);
        expect(result.result.hasMore).toBe(true);
      });

      it('should handle pagination', async () => {
        const mockResponse = {
          object: 'list',
          results: [],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'list_users',
          startCursor: 'cursor_token',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
      });
    });

    describe('retrieveUser', () => {
      it('should retrieve user successfully', async () => {
        const mockResponse = {
          id: 'user_123',
          name: 'John Doe',
          avatar_url: 'https://example.com/avatar.jpg',
          type: 'person',
          person: {
            email: 'john@example.com',
          },
          object: 'user',
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'retrieve_user',
          userId: 'user_123',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.user.id).toBe('user_123');
        expect(result.result.user.name).toBe('John Doe');
        expect(result.result.user.person.email).toBe('john@example.com');
      });

      it('should handle non-existent user', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: false,
          status: 404,
          json: async () => ({
            message: 'Not found',
          }),
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'retrieve_user',
          userId: 'user_nonexistent',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(false);
      });
    });
  });

  describe('Search Operations', () => {
    describe('search', () => {
      it('should search pages successfully', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'page_1',
              object: 'page',
              properties: {
                title: {
                  title: [
                    {
                      text: {
                        content: 'Project Plan',
                      },
                    },
                  ],
                },
              },
            },
            {
              id: 'page_2',
              object: 'page',
              properties: {
                title: {
                  title: [
                    {
                      text: {
                        content: 'Meeting Notes',
                      },
                    },
                  ],
                },
              },
            },
          ],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'search',
          query: 'project',
          filter: {
            property: 'object',
            value: 'page',
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(2);
      });

      it('should search databases', async () => {
        const mockResponse = {
          object: 'list',
          results: [
            {
              id: 'database_1',
              object: 'database',
              title: [
                {
                  type: 'text',
                  text: {
                    content: 'Tasks Database',
                  },
                },
              ],
            },
          ],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'search',
          query: 'tasks',
          filter: {
            property: 'object',
            value: 'database',
          },
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(1);
        expect(result.result.results[0].object).toBe('database');
      });

      it('should handle empty search results', async () => {
        const mockResponse = {
          object: 'list',
          results: [],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'search',
          query: 'nonexistent',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.results).toHaveLength(0);
      });

      it('should handle pagination in search', async () => {
        const mockResponse = {
          object: 'list',
          results: [],
          next_cursor: 'next_cursor',
          has_more: true,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        notionBubble = new NotionBubble({
          operation: 'search',
          query: 'test',
          startCursor: 'cursor_token',
          credentials: mockCredentials,
        });

        const result = await notionBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.hasMore).toBe(true);
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle authentication errors', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({
          message: 'Unauthorized',
        }),
      } as Response);

      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
        credentials: { [CredentialType.NOTION_CRED]: 'invalid_key' },
      });

      const result = await notionBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('Unauthorized');
    });

    it('should handle rate limiting', async () => {
      vi.mocked(fetch)
        .mockRejectedValueOnce(new Error('Rate limit exceeded'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'page_test_123',
            object: 'page',
            properties: {},
          }),
        } as Response);

      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2);
    });

    it('should handle network errors', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));

      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('Network error');
    });

    it('should handle malformed responses', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          invalid: 'response',
        }),
      } as Response);

      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.result.success).toBe(false);
    });
  });

  describe('Credential Testing', () => {
    it('should test credentials successfully', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'user_test',
          object: 'user',
        }),
      } as Response);

      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
        credentials: mockCredentials,
      });

      const isValid = await notionBubble.testCredential();

      expect(isValid).toBe(true);
    });

    it('should handle invalid credentials', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
      } as Response);

      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
        credentials: { [CredentialType.NOTION_CRED]: 'invalid_key' },
      });

      const isValid = await notionBubble.testCredential();

      expect(isValid).toBe(false);
    });

    it('should handle missing credentials', async () => {
      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
      });

      const isValid = await notionBubble.testCredential();

      expect(isValid).toBe(false);
    });
  });

  describe('Retry Logic', () => {
    it('should retry on transient errors', async () => {
      vi.mocked(fetch)
        .mockRejectedValueOnce(new Error('ECONNRESET'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'page_test_123',
            object: 'page',
            properties: {},
          }),
        } as Response);

      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2);
    });

    it('should respect max retry limit', async () => {
      vi.mocked(fetch).mockRejectedValue(new Error('Persistent error'));

      notionBubble = new NotionBubble({
        operation: 'retrievePage',
        pageId: 'page_test_123',
        credentials: mockCredentials,
      });

      const result = await notionBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(4); // Initial + 3 retries
    });
  });
});
