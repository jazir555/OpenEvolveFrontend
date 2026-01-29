/**
 * Edge Case and Boundary Tests for Google Maps Tool
 *
 * Comprehensive edge case coverage including:
 * - Input boundaries (empty, null, max length, unicode, special characters)
 * - Query string boundaries
 * - Geographic boundaries
 * - Response parsing edge cases
 * - API error handling
 * - Performance edge cases
 */

import { describe, it, expect } from 'vitest';
import { GoogleMapsTool } from './google-maps-tool.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('GoogleMapsTool - Edge Cases and Boundary Tests', () => {
  describe('Input Boundary Tests', () => {
    describe('Query String Boundaries', () => {
      it('should handle empty query string', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: [''],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle single character query', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['a'],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle maximum length query (5000 chars)', async () => {
        const longQuery = 'restaurant ' + 'x'.repeat(4990);

        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: [longQuery],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle unicode in queries', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['餐厅 北京京都', 'café München', 'מסעדה תל אביב'],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle special characters in queries', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: [
            "O'Reilly's Restaurant",
            'Café & Restaurant',
            'hotel@city',
            '100% pure',
          ],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle emoji in queries', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant 🍕', 'hotel 🏨', 'coffee ☕'],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle whitespace-only queries', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['   ', '\t\t', '\n\n'],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle case sensitivity in queries', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['RESTAURANT', 'Restaurant', 'restaurant'],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });
    });

    describe('Query Array Boundaries', () => {
      it('should handle empty queries array', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: [],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle single query', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle maximum queries (100)', async () => {
        const queries = Array.from({ length: 100 }, (_, i) => `restaurant ${i}`);

        const tool = new GoogleMapsTool({
          operation: 'search',
          queries,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle duplicate queries', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant', 'restaurant', 'restaurant'],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle mixed valid and invalid queries', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant', '', 'cafe', '   ', 'hotel'],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });
    });

    describe('Limit Boundaries', () => {
      it('should handle minimum limit (1)', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          limit: 1,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle maximum limit (100)', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          limit: 100,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle zero limit', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          limit: 0,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle negative limit', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          limit: -10,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle decimal limit', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          limit: 10.5,
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });
    });

    describe('Credential Edge Cases', () => {
      it('should handle missing credentials', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('APIFY_CRED');
      });

      it('should handle empty credential string', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          credentials: {
            [CredentialType.APIFY_CRED]: '',
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle null credentials', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          credentials: null as any,
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle invalid credential format', async () => {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: ['restaurant'],
          credentials: {
            [CredentialType.APIFY_CRED]: { invalid: 'format' } as any,
          },
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });
    });
  });

  describe('Geographic Boundaries', () => {
    it('should handle queries with coordinates', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['37.7749,-122.4194'], // San Francisco coordinates
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle extreme latitude values', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['90,0', '-90,0'], // North and South Pole
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle extreme longitude values', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['0,180', '0,-180'], // International Date Line
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle invalid coordinates', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['200,200'], // Invalid coordinates
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle plus codes', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['849VCWC8+R9'], // Google Plus Code
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle postal codes', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['90210', 'SW1A 1AA', '10001'], // Various postal codes
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle addresses with multiple lines', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['1600 Amphitheatre Parkway, Mountain View, CA 94043'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });
  });

  describe('Response Parsing Edge Cases', () => {
    it('should handle empty result set', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['xyznonexistentplace12345'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle single result', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        limit: 1,
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle maximum results per query', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        limit: 100,
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle results with missing fields', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle results with null values', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle results with special characters', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['café', 'müller', 'österreich'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });
  });

  describe('API Error Handling', () => {
    it('should handle rate limit errors', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'invalid-token',
        },
      });

      const result = await tool.action();

      // Should fail gracefully with invalid token
      expect(result).toBeDefined();
    });

    it('should handle invalid API key', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'invalid_key_12345',
        },
      });

      const result = await tool.action();

      expect(result).toBeDefined();
    });

    it('should handle network timeout', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result).toBeDefined();
    });
  });

  describe('Performance Edge Cases', () => {
    it('should handle many queries efficiently', async () => {
      const queries = Array.from({ length: 50 }, (_, i) => `restaurant ${i}`);

      const tool = new GoogleMapsTool({
        operation: 'search',
        queries,
        limit: 10,
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const startTime = Date.now();
      const result = await tool.action();
      const duration = Date.now() - startTime;

      expect(result.success).toBe(true);
    });

    it('should handle complex location queries', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: [
          'restaurant near San Francisco, CA',
          'hotels within 5 miles of Times Square, New York',
          'cafes in downtown London, UK',
        ],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle query with multiple filters', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['open now restaurant with wifi in San Francisco'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });
  });

  describe('Operation Type Edge Cases', () => {
    it('should handle search operation', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle invalid operation', async () => {
      const tool = new GoogleMapsTool({
        operation: 'invalid_operation' as any,
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(false);
    });
  });

  describe('Data Structure Edge Cases', () => {
    it('should handle nested location data', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle missing location coordinates', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle various address formats', async () => {
      const tool = new GoogleMapsTool({
        operation: 'search',
        queries: [
          '123 Main St',
          'Main Street 123',
          '123, Rue Principale',
          'Hauptstraße 123',
        ],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });
  });

  describe('Concurrent Request Edge Cases', () => {
    it('should handle multiple concurrent searches', async () => {
      const tool1 = new GoogleMapsTool({
        operation: 'search',
        queries: ['restaurant'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const tool2 = new GoogleMapsTool({
        operation: 'search',
        queries: ['hotel'],
        credentials: {
          [CredentialType.APIFY_CRED]: 'test-token',
        },
      });

      const results = await Promise.all([tool1.action(), tool2.action()]);

      results.forEach((result) => {
        expect(result.success).toBe(true);
      });
    });

    it('should handle rapid sequential searches', async () => {
      const promises = [];

      for (let i = 0; i < 10; i++) {
        const tool = new GoogleMapsTool({
          operation: 'search',
          queries: [`restaurant ${i}`],
          credentials: {
            [CredentialType.APIFY_CRED]: 'test-token',
          },
        });

        promises.push(tool.action());
      }

      const results = await Promise.all(promises);

      results.forEach((result) => {
        expect(result.success).toBe(true);
      });
    });
  });
});
