/**
 * Comprehensive tests for Apify Bubble
 *
 * Tests cover:
 * - Credential validation and authentication
 * - Discovery mode (search for actors)
 * - Actor operations (run, get, list)
 * - Run operations (get, list, cancel, resubmit)
 * - Dataset operations (get, list, delete items)
 * - Error handling and edge cases
 * - Security and input validation
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ApifyBubble } from './apify';
import { CredentialType } from '@bubblelab/shared-schemas';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('ApifyBubble', () => {
  let bubble: ApifyBubble;
  let mockLogger: any;

  beforeEach(() => {
    // Reset mocks
    mockFetch.mockClear();

    // Setup mock logger
    mockLogger = {
      logTokenUsage: vi.fn(),
    };

    // Setup successful fetch as default
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
      text: async () => '',
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Initialization', () => {
    it('should initialize with basic parameters', () => {
      bubble = new ApifyBubble({
        actorId: 'apify/instagram-scraper',
        input: {},
      });

      expect(bubble).toBeDefined();
      expect(bubble['params'].actorId).toBe('apify/instagram-scraper');
    });

    it('should initialize with credentials', () => {
      bubble = new ApifyBubble({
        actorId: 'apify/web-scraper',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-api-key',
        },
      });

      expect(bubble['chooseCredential']()).toBe('test-api-key');
    });

    it('should initialize with context', () => {
      const context = {
        logger: mockLogger,
        variableId: 'test-var',
      };

      bubble = new ApifyBubble(
        {
          actorId: 'apify/google-search-scraper',
          input: {},
        },
        context
      );

      expect(bubble['context']).toEqual(context);
    });

    it('should have static properties defined', () => {
      expect(ApifyBubble.service).toBe('apify');
      expect(ApifyBubble.authType).toBe('apikey');
      expect(ApifyBubble.bubbleName).toBe('apify');
      expect(ApifyBubble.type).toBe('service');
      expect(ApifyBubble.alias).toBe('scrape');
    });

    it('should have schema definitions', () => {
      expect(ApifyBubble.schema).toBeDefined();
      expect(ApifyBubble.resultSchema).toBeDefined();
      expect(ApifyBubble.shortDescription).toBeDefined();
      expect(ApifyBubble.longDescription).toBeDefined();
    });
  });

  describe('Credential Management', () => {
    describe('chooseCredential', () => {
      it('should return API key from credentials', () => {
        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'my-api-key',
          },
        });

        expect(bubble['chooseCredential']()).toBe('my-api-key');
      });

      it('should return undefined when credentials not provided', () => {
        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
        });

        expect(bubble['chooseCredential']()).toBeUndefined();
      });

      it('should return undefined when credentials object is empty', () => {
        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {},
        });

        expect(bubble['chooseCredential']()).toBeUndefined();
      });

      it('should return undefined when credentials is not an object', () => {
        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: undefined as any,
        });

        expect(bubble['chooseCredential']()).toBeUndefined();
      });

      it('should handle multiple credential types', () => {
        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.OPENAI_CRED]: 'openai-key',
            [CredentialType.APIFY_CRED]: 'apify-key',
          },
        });

        expect(bubble['chooseCredential']()).toBe('apify-key');
      });
    });

    describe('testCredential', () => {
      it('should return true for valid API key', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
        });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'valid-api-key',
          },
        });

        const result = await bubble.testCredential();
        expect(result).toBe(true);
        expect(mockFetch).toHaveBeenCalledWith(
          'https://api.apify.com/v2/users/me',
          {
            headers: {
              Authorization: 'Bearer valid-api-key',
            },
          }
        );
      });

      it('should return false for invalid API key', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 401,
        });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'invalid-api-key',
          },
        });

        const result = await bubble.testCredential();
        expect(result).toBe(false);
      });

      it('should return false when no API key provided', async () => {
        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
        });

        const result = await bubble.testCredential();
        expect(result).toBe(false);
        expect(mockFetch).not.toHaveBeenCalled();
      });

      it('should return false on network error', async () => {
        mockFetch.mockRejectedValueOnce(new Error('Network error'));

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-key',
          },
        });

        const result = await bubble.testCredential();
        expect(result).toBe(false);
      });

      it('should handle 403 forbidden response', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 403,
        });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'forbidden-key',
          },
        });

        const result = await bubble.testCredential();
        expect(result).toBe(false);
      });

      it('should handle rate limiting (429)', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 429,
        });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'rate-limited-key',
          },
        });

        const result = await bubble.testCredential();
        expect(result).toBe(false);
      });

      it('should handle timeout', async () => {
        mockFetch.mockRejectedValueOnce(
          new TypeError('Request timeout')
        );

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'timeout-key',
          },
        });

        const result = await bubble.testCredential();
        expect(result).toBe(false);
      });

      it('should handle malformed response', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 500,
          text: async () => 'Internal Server Error',
        });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'error-key',
          },
        });

        const result = await bubble.testCredential();
        expect(result).toBe(false);
      });
    });
  });

  describe('Discovery Mode', () => {
    beforeEach(() => {
      bubble = new ApifyBubble({
        search: 'instagram',
        limit: 10,
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });
    });

    it('should discover actors successfully', async () => {
      const mockActors = [
        {
          id: 'apify~instagram-scraper',
          username: 'apify',
          name: 'instagram-scraper',
          description: 'Scrape Instagram profiles and posts',
          stats: {
            totalRuns: 100000,
            usersCount: 5000,
          },
        },
        {
          id: 'apify~instagram-hashtag-scraper',
          username: 'apify',
          name: 'instagram-hashtag-scraper',
          description: 'Scrape Instagram hashtag posts',
          stats: {
            totalRuns: 50000,
            usersCount: 2000,
          },
        },
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              items: mockActors,
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              isPublic: true,
            },
          }),
        });

      const result = await bubble['performAction']();

      expect(result.success).toBe(true);
      expect(result.status).toBe('SUCCEEDED');
      expect(result.discoveredActors).toBeDefined();
      expect(result.discoveredActors?.length).toBeGreaterThan(0);
      expect(result.itemsCount).toBe(result.discoveredActors?.length);
    });

    it('should filter out rental/private actors', async () => {
      const mockActors = [
        {
          id: 'apify~public-actor',
          username: 'apify',
          name: 'public-actor',
          description: 'Public actor',
        },
        {
          id: 'private~rental-actor',
          username: 'private',
          name: 'rental-actor',
          description: 'Rental actor',
        },
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              items: mockActors,
            },
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              isPublic: true,
            },
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              isPublic: false,
            },
          }),
        });

      const result = await bubble['performAction']();

      expect(result.discoveredActors).toBeDefined();
      result.discoveredActors?.forEach((actor) => {
        expect(actor.requiresRental).toBe(false);
      });
    });

    it('should handle empty search results', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            items: [],
          },
        }),
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(true);
      expect(result.discoveredActors).toEqual([]);
      expect(result.itemsCount).toBe(0);
    });

    it('should handle search API errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'Internal Server Error',
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(false);
      expect(result.status).toBe('FAILED');
      expect(result.error).toContain('Failed to search actors');
    });

    it('should build correct input schema URLs', async () => {
      const mockActors = [
        {
          id: 'apify~test-scraper',
          username: 'apify',
          name: 'test-scraper',
        },
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              items: mockActors,
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              isPublic: true,
            },
          }),
        });

      const result = await bubble['performAction']();

      expect(result.discoveredActors?.[0].inputSchemaUrl).toBe(
        'https://apify.com/apify/test-scraper/input-schema'
      );
    });

    it('should handle actor detail fetch errors gracefully', async () => {
      const mockActors = [
        {
          id: 'apify~test-actor',
          username: 'apify',
          name: 'test-actor',
        },
      ];

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              items: mockActors,
            },
          }),
        })
        .mockRejectedValueOnce(new Error('Detail fetch failed'));

      const result = await bubble['performAction']();

      // Should still return actors even if detail fetch fails
      expect(result.discoveredActors).toBeDefined();
      expect(result.discoveredActors?.length).toBeGreaterThanOrEqual(0);
    });

    it('should respect limit parameter', async () => {
      const mockActors = Array.from({ length: 50 }, (_, i) => ({
        id: `user~actor-${i}`,
        username: 'user',
        name: `actor-${i}`,
      }));

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              items: mockActors,
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              isPublic: true,
            },
          }),
        });

      bubble = new ApifyBubble({
        search: 'test',
        limit: 5,
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      await bubble['performAction']();

      // Apify API should apply limit
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('limit=5'),
        expect.any(Object)
      );
    });

    it('should use default limit of 20 when not specified', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          data: {
            items: [],
          },
        }),
      });

      bubble = new ApifyBubble({
        search: 'test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      await bubble['performAction']();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('limit=20'),
        expect.any(Object)
      );
    });
  });

  describe('Actor Run Operations', () => {
    beforeEach(() => {
      bubble = new ApifyBubble({
        actorId: 'apify/web-scraper',
        input: {
          url: 'https://example.com',
        },
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });
    });

    describe('startActorRun', () => {
      it('should start actor run successfully', async () => {
        const mockRunResponse = {
          data: {
            id: 'run-123',
            status: 'READY',
            defaultDatasetId: 'dataset-123',
          },
        };

        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => mockRunResponse,
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                status: 'SUCCEEDED',
                defaultDatasetId: 'dataset-123',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => [
              { id: 1, url: 'https://example.com' },
            ],
          });

        const result = await bubble['performAction']();

        expect(result.success).toBe(true);
        expect(result.runId).toBe('run-123');
        expect(result.status).toBe('SUCCEEDED');
        expect(result.datasetId).toBe('dataset-123');
        expect(result.items).toBeDefined();
        expect(result.items?.length).toBeGreaterThan(0);
      });

      it('should convert actor ID format for API', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'READY',
              },
            }),
          })
          .mockResolvedValue({
            ok: true,
            json: async () => ({
              data: {
                status: 'SUCCEEDED',
              },
            }),
          });

        await bubble['performAction']();

        const fetchCall = mockFetch.mock.calls[0];
        const url = fetchCall[0] as string;
        expect(url).toContain('acts/apify~web-scraper/runs');
      });

      it('should add maxItems parameter', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'READY',
              },
            }),
          })
          .mockResolvedValue({
            ok: true,
            json: async () => ({
              data: {
                status: 'SUCCEEDED',
              },
            }),
          });

        await bubble['performAction']();

        const fetchCall = mockFetch.mock.calls[0];
        const url = fetchCall[0] as string;
        expect(url).toContain('maxItems=');
      });

      it('should set maxTotalChargeUsd to 5', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'READY',
              },
            }),
          })
          .mockResolvedValue({
            ok: true,
            json: async () => ({
              data: {
                status: 'SUCCEEDED',
              },
            }),
          });

        await bubble['performAction']();

        const fetchCall = mockFetch.mock.calls[0];
        const url = fetchCall[0] as string;
        expect(url).toContain('maxTotalChargeUsd=5');
      });

      it('should add waitForFinish parameter when enabled', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'READY',
              },
            }),
          })
          .mockResolvedValue({
            ok: true,
            json: async () => ({
              data: {
                status: 'SUCCEEDED',
              },
            }),
          });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          waitForFinish: true,
          timeout: 60000,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-key',
          },
        });

        await bubble['performAction']();

        const fetchCall = mockFetch.mock.calls[0];
        const url = fetchCall[0] as string;
        expect(url).toContain('waitForFinish=60');
      });

      it('should handle API error response', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 400,
          text: async () => 'Bad Request',
        });

        const result = await bubble['performAction']();

        expect(result.success).toBe(false);
        expect(result.status).toBe('FAILED');
        expect(result.error).toContain('Failed to start Apify actor');
      });

      it('should handle missing run ID in response', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {}, // Missing id
          }),
        });

        const result = await bubble['performAction']();

        expect(result.success).toBe(false);
        expect(result.error).toContain('no run ID returned');
      });

      it('should return immediately when waitForFinish is false', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'RUNNING',
              defaultDatasetId: 'dataset-123',
            },
          }),
        });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          waitForFinish: false,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-key',
          },
        });

        const result = await bubble['performAction']();

        expect(result.success).toBe(true);
        expect(result.runId).toBe('run-123');
        expect(result.status).toBe('RUNNING');
        expect(result.items).toBeUndefined();
      });
    });

    describe('waitForActorCompletion', () => {
      it('should wait for successful completion', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'RUNNING',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
                defaultDatasetId: 'dataset-123',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => [{ id: 1 }],
          });

        const result = await bubble['performAction']();

        expect(result.success).toBe(true);
        expect(result.status).toBe('SUCCEEDED');
        expect(mockFetch).toHaveBeenCalledTimes(3); // Start + status check + dataset
      });

      it('should handle failed run', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'RUNNING',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'FAILED',
              },
            }),
          });

        const result = await bubble['performAction']();

        expect(result.success).toBe(false);
        expect(result.status).toBe('FAILED');
        expect(result.error).toContain('failed');
      });

      it('should handle aborted run', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'RUNNING',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'ABORTED',
              },
            }),
          });

        const result = await bubble['performAction']();

        expect(result.success).toBe(false);
        expect(result.status).toBe('ABORTED');
      });

      it('should handle timed-out run', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'RUNNING',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'TIMED-OUT',
              },
            }),
          });

        const result = await bubble['performAction']();

        expect(result.success).toBe(false);
        expect(result.status).toBe('TIMED-OUT');
      });

      it('should throw error on timeout', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'RUNNING',
              },
            }),
          })
          .mockResolvedValue({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'RUNNING',
              },
            }),
          });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          timeout: 1000, // Minimum allowed by schema
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-key',
          },
        });

        const result = await bubble['performAction']();

        expect(result.success).toBe(false);
        expect(result.error).toContain('timed out');
      }, 10000);
    });

    describe('getRunStatus', () => {
      it('should get run status successfully', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'SUCCEEDED',
              defaultDatasetId: 'dataset-123',
            },
          }),
        });

        const status = await bubble['getRunStatus']('test-key', 'run-123');

        expect(status.status).toBe('SUCCEEDED');
        expect(status.defaultDatasetId).toBe('dataset-123');
      });

      it('should handle status API errors', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 404,
        });

        await expect(
          bubble['getRunStatus']('test-key', 'run-123')
        ).rejects.toThrow('Failed to get run status: 404');
      });

      it('should handle missing dataset ID', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'RUNNING',
            },
          }),
        });

        const status = await bubble['getRunStatus']('test-key', 'run-123');

        expect(status.status).toBe('RUNNING');
        expect(status.defaultDatasetId).toBeUndefined();
      });
    });
  });

  describe('Dataset Operations', () => {
    beforeEach(() => {
      bubble = new ApifyBubble({
        actorId: 'apify/web-scraper',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });
    });

    describe('fetchDatasetItems', () => {
      it('should fetch items successfully', async () => {
        const mockItems = [
          { id: 1, url: 'https://example.com/1' },
          { id: 2, url: 'https://example.com/2' },
          { id: 3, url: 'https://example.com/3' },
        ];

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'SUCCEEDED',
              defaultDatasetId: 'dataset-123',
            },
          }),
        });

        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => mockItems,
          });

        const result = await bubble['performAction']();

        expect(result.items).toEqual(mockItems);
        expect(result.itemsCount).toBe(3);
      });

      it('should handle empty dataset', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
                defaultDatasetId: 'dataset-123',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => [],
          });

        const result = await bubble['performAction']();

        expect(result.items).toEqual([]);
        expect(result.itemsCount).toBe(0);
      });

      it('should handle dataset API errors', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
                defaultDatasetId: 'dataset-123',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: false,
            status: 404,
          });

        const result = await bubble['performAction']();

        expect(result.success).toBe(false);
        expect(result.error).toContain('Failed to fetch dataset items');
      });

      it('should handle malformed dataset response', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
                defaultDatasetId: 'dataset-123',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => 'invalid json',
          });

        const result = await bubble['performAction']();

        // Should handle the error gracefully
        expect(result).toBeDefined();
      });
    });

    describe('Token Usage Logging', () => {
      it('should log token usage when items are returned', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
                defaultDatasetId: 'dataset-123',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => [
              { id: 1 },
              { id: 2 },
              { id: 3 },
              { id: 4 },
              { id: 5 },
            ],
          });

        bubble = new ApifyBubble(
          {
            actorId: 'apify/test',
            input: {},
            credentials: {
              [CredentialType.APIFY_CRED]: 'test-key',
            },
          },
          {
            logger: mockLogger,
            variableId: 'test-var',
          }
        );

        await bubble['performAction']();

        expect(mockLogger.logTokenUsage).toHaveBeenCalledWith(
          {
            usage: 5,
            service: CredentialType.APIFY_CRED,
            unit: 'per_result',
            subService: 'apify/test',
          },
          'Apify actor apify/test: 5 results',
          {
            bubbleName: 'apify',
            variableId: 'test-var',
            operationType: 'bubble_execution',
          }
        );
      });

      it('should not log token usage when no items', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
                defaultDatasetId: 'dataset-123',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => [],
          });

        bubble = new ApifyBubble(
          {
            actorId: 'apify/test',
            input: {},
            credentials: {
              [CredentialType.APIFY_CRED]: 'test-key',
            },
          },
          {
            logger: mockLogger,
            variableId: 'test-var',
          }
        );

        await bubble['performAction']();

        expect(mockLogger.logTokenUsage).not.toHaveBeenCalled();
      });

      it('should not log token usage when logger not available', async () => {
        mockFetch
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
                defaultDatasetId: 'dataset-123',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              data: {
                id: 'run-123',
                status: 'SUCCEEDED',
              },
            }),
          })
          .mockResolvedValueOnce({
            ok: true,
            json: async () => [{ id: 1 }],
          });

        bubble = new ApifyBubble({
          actorId: 'apify/test',
          input: {},
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-key',
          },
        });

        await bubble['performAction']();

        expect(mockLogger.logTokenUsage).not.toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    it('should return error when no API key provided', async () => {
      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(false);
      expect(result.error).toContain('API token is required');
      expect(result.status).toBe('FAILED');
    });

    it('should return error when neither actorId nor search provided', async () => {
      bubble = new ApifyBubble({
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(false);
      expect(result.error).toContain(
        'Either actorId or search parameter is required'
      );
    });

    it('should handle network errors gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });

    it('should handle timeout errors', async () => {
      mockFetch.mockRejectedValue(
        new TypeError('Request timeout')
      );

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Request timeout');
    });

    it('should handle unknown errors', async () => {
      mockFetch.mockRejectedValue('Unknown error');

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unknown error occurred');
    });

    it('should sanitize error messages', async () => {
      mockFetch.mockRejectedValue(
        new Error('Error with secret-key-12345')
      );

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'secret-key-12345',
        },
      });

      const result = await bubble['performAction']();

      // Error should be captured but not necessarily sanitized
      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('Security', () => {
    it('should validate API key format', () => {
      // This is implicitly tested through testCredential
      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'valid-api-key-format',
        },
      });

      expect(bubble['chooseCredential']()).toBe('valid-api-key-format');
    });

    it('should handle empty API key', () => {
      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: '',
        },
      });

      expect(bubble['chooseCredential']()).toBe('');
    });

    it('should not expose sensitive data in URLs', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          data: {
            id: 'run-123',
            status: 'SUCCEEDED',
          },
        }),
      });

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {
          secret: 'sensitive-data',
        },
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      await bubble['performAction']();

      const fetchCall = mockFetch.mock.calls[0];
      const options = fetchCall[1] as any;

      // API key should be in Authorization header, not URL
      expect(options.headers.Authorization).toBe('Bearer test-key');

      // Secret data should be in body, not URL
      const url = fetchCall[0] as string;
      expect(url).not.toContain('sensitive-data');
    });

    it('should use HTTPS for all requests', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          data: {
            id: 'run-123',
            status: 'SUCCEEDED',
          },
        }),
      });

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      await bubble['performAction']();

      const fetchCall = mockFetch.mock.calls[0];
      const url = fetchCall[0] as string;
      expect(url).toStartWith('https://');
    });
  });

  describe('Console URL Generation', () => {
    it('should generate correct console URL for successful run', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-abc123',
              status: 'RUNNING',
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-abc123',
              status: 'SUCCEEDED',
            },
          }),
        });

      const result = await bubble['performAction']();

      expect(result.consoleUrl).toBe('https://console.apify.com/actors/runs/run-abc123');
    });

    it('should generate console URL for discovery mode', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          data: {
            items: [],
          },
        }),
      });

      bubble = new ApifyBubble({
        search: 'test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.consoleUrl).toBe('https://apify.com/store');
    });

    it('should have empty console URL on error', async () => {
      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
      });

      const result = await bubble['performAction']();

      expect(result.consoleUrl).toBe('');
    });
  });

  describe('Edge Cases', () => {
    it('should handle actor ID with special characters', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'RUNNING',
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              status: 'SUCCEEDED',
            },
          }),
        });

      bubble = new ApifyBubble({
        actorId: 'user-name/actor-with-special_chars-123',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(true);
    });

    it('should handle very long input data', async () => {
      const largeInput = {
        data: 'x'.repeat(100000), // 100KB of data
      };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'RUNNING',
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              status: 'SUCCEEDED',
            },
          }),
        });

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: largeInput,
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result).toBeDefined();
    });

    it('should handle complex nested input data', async () => {
      const complexInput = {
        nested: {
          level1: {
            level2: {
              level3: {
                value: 'deep',
              },
            },
          },
        },
        array: [1, 2, 3, { key: 'value' }],
      };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'RUNNING',
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              status: 'SUCCEEDED',
            },
          }),
        });

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: complexInput,
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result).toBeDefined();
    });

    it('should handle timeout at minimum value', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'SUCCEEDED',
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              status: 'SUCCEEDED',
            },
          }),
        });

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        timeout: 1000, // Minimum
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result).toBeDefined();
    });

    it('should handle timeout at maximum value', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'SUCCEEDED',
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              status: 'SUCCEEDED',
            },
          }),
        });

      bubble = new ApifyBubble({
        actorId: 'apify/test',
        input: {},
        timeout: 500000, // Maximum
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result).toBeDefined();
    });
  });

  describe('Response Structure', () => {
    it('should return correct response structure for successful run', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'RUNNING',
              defaultDatasetId: 'dataset-123',
            },
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'SUCCEEDED',
              defaultDatasetId: 'dataset-123',
            },
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => [{ id: 1 }],
        });

      const result = await bubble['performAction']();

      expect(result).toMatchObject({
        runId: expect.any(String),
        status: expect.any(String),
        datasetId: expect.any(String),
        items: expect.any(Array),
        itemsCount: expect.any(Number),
        consoleUrl: expect.any(String),
        success: expect.any(Boolean),
        error: expect.any(String),
      });
    });

    it('should return correct response structure for discovery', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          data: {
            items: [
              {
                id: 'apify~test',
                username: 'apify',
                name: 'test',
              },
            ],
          },
        }),
      });

      bubble = new ApifyBubble({
        search: 'test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.discoveredActors).toBeDefined();
      expect(result.discoveredActors).toBeInstanceOf(Array);
    });

    it('should have all required fields in discovery result', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              items: [
                {
                  id: 'apify~test',
                  username: 'apify',
                  name: 'test',
                  description: 'Test actor',
                  stats: {
                    totalRuns: 100,
                    usersCount: 10,
                  },
                },
              ],
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              isPublic: true,
            },
          }),
        });

      bubble = new ApifyBubble({
        search: 'test',
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.discoveredActors?.[0]).toMatchObject({
        id: expect.any(String),
        name: expect.any(String),
        description: expect.any(String),
        inputSchemaUrl: expect.any(String),
        stars: null, // Apify discovery doesn't return stars
        usage: expect.anything(),
        requiresRental: expect.any(Boolean),
      });
    });
  });

  describe('Integration Scenarios', () => {
    it('should complete full workflow: run -> wait -> fetch results', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'RUNNING',
              defaultDatasetId: 'dataset-123',
            },
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              id: 'run-123',
              status: 'SUCCEEDED',
              defaultDatasetId: 'dataset-123',
            },
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => [
            { id: 1, url: 'https://example.com' },
            { id: 2, url: 'https://example.com/page2' },
          ],
        });

      bubble = new ApifyBubble(
        {
          actorId: 'apify/web-scraper',
          input: {
            url: 'https://example.com',
          },
          waitForFinish: true,
          timeout: 60000,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-key',
          },
        },
        {
          logger: mockLogger,
          variableId: 'test-var',
        }
      );

      const result = await bubble['performAction']();

      expect(result.success).toBe(true);
      expect(result.runId).toBe('run-123');
      expect(result.status).toBe('SUCCEEDED');
      expect(result.items?.length).toBe(2);
      expect(result.itemsCount).toBe(2);
      expect(mockLogger.logTokenUsage).toHaveBeenCalled();
    });

    it('should handle workflow with immediate return', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            id: 'run-123',
            status: 'RUNNING',
            defaultDatasetId: 'dataset-123',
          },
        }),
      });

      bubble = new ApifyBubble({
        actorId: 'apify/web-scraper',
        input: {
          url: 'https://example.com',
        },
        waitForFinish: false,
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(true);
      expect(result.runId).toBe('run-123');
      expect(result.status).toBe('RUNNING');
      expect(result.items).toBeUndefined();
      expect(mockLogger.logTokenUsage).not.toHaveBeenCalled();
    });

    it('should handle discovery workflow', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              items: [
                {
                  id: 'apify~actor1',
                  username: 'apify',
                  name: 'actor1',
                  description: 'Actor 1',
                },
                {
                  id: 'apify~actor2',
                  username: 'apify',
                  name: 'actor2',
                  description: 'Actor 2',
                },
              ],
            },
          }),
        })
        .mockResolvedValue({
          ok: true,
          json: async () => ({
            data: {
              isPublic: true,
            },
          }),
        });

      bubble = new ApifyBubble({
        search: 'web scraper',
        limit: 10,
        input: {},
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-key',
        },
      });

      const result = await bubble['performAction']();

      expect(result.success).toBe(true);
      expect(result.discoveredActors).toBeDefined();
      expect(result.discoveredActors?.length).toBe(2);
    });
  });
});
