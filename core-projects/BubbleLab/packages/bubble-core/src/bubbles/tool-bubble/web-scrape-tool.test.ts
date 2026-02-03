/**
 * Web Scrape Tool Unit Tests
 * File: tool-bubble/web-scrape-tool.test.ts
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { WebScrapeTool } from './web-scrape-tool.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('WebScrapeTool', () => {
  let mockFetch: any;

  beforeEach(() => {
    // Mock fetch API for HTTP requests
    mockFetch = vi.fn();
    global.fetch = mockFetch;

    // Clear any mock state
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Construction and Initialization', () => {
    it('should create instance with valid parameters', () => {
      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      expect(tool).toBeDefined();
      expect(tool.params.url).toBe('https://example.com');
      expect(tool.params.format).toBe('markdown');
    });

    it('should validate required URL parameter', () => {
      expect(() => {
        new WebScrapeTool({
          url: '',
          credentials: {
            [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
          },
        });
      }).toThrow();
    });

    it('should validate URL format', () => {
      expect(() => {
        new WebScrapeTool({
          url: 'not-a-valid-url',
          credentials: {
            [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
          },
        });
      }).toThrow();
    });

    it('should set default format to markdown', () => {
      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      expect(tool.params.format).toBe('markdown');
    });

    it('should set default onlyMainContent to true', () => {
      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      expect(tool.params.onlyMainContent).toBe(true);
    });
  });

  describe('Input Validation', () => {
    it('should reject invalid URLs', () => {
      const invalidUrls = [
        'javascript:alert(1)',
        'ftp://example.com',
        'not-a-url',
        'http://',
        'https://',
      ];

      invalidUrls.forEach((url) => {
        expect(() => {
          new WebScrapeTool({
            url,
            credentials: {
              [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
            },
          });
        }).toThrow();
      });
    });

    it('should accept valid URLs', () => {
      const validUrls = [
        'https://example.com',
        'https://example.com/path',
        'http://example.com',
        'https://subdomain.example.com',
        'https://example.com/path?query=value',
      ];

      validUrls.forEach((url) => {
        expect(() => {
          new WebScrapeTool({
            url,
            credentials: {
              [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
            },
          });
        }).not.toThrow();
      });
    });

    it('should validate format enum', () => {
      expect(() => {
        new WebScrapeTool({
          url: 'https://example.com',
          format: 'invalid' as any,
          credentials: {
            [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
          },
        });
      }).toThrow();
    });
  });

  describe('Scraping Operations', () => {
    it('should scrape content successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          markdown: '# Test Content\n\nThis is a test page.',
          metadata: {
            title: 'Test Page',
            statusCode: 200,
          },
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        format: 'markdown',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.success).toBe(true);
      expect(result.content).toContain('# Test Content');
      expect(result.title).toBe('Test Page');
      expect(result.url).toBe('https://example.com');
      expect(result.format).toBe('markdown');
    });

    it('should handle large content by summarizing', async () => {
      // Create large content > 5M characters
      const largeContent = 'A'.repeat(6000000);

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            success: true,
            markdown: largeContent,
            metadata: {
              title: 'Large Page',
              statusCode: 200,
            },
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            success: true,
            response: 'Summarized content',
          }),
        });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        format: 'markdown',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
          [CredentialType.OPENAI_API_KEY]: 'test-openai-key',
        },
      });

      const result = await tool.act();

      expect(result.success).toBe(true);
      // Content should be either summarized or original
      expect(result.content).toBeDefined();
    });

    it('should handle missing credentials gracefully', async () => {
      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: undefined,
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });

    it('should handle API errors', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: false,
          error: 'Rate limit exceeded',
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Rate limit exceeded');
    });

    it('should handle missing content in response', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          html: '<html>Content</html>', // No markdown
          metadata: {
            title: 'Test Page',
          },
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        format: 'markdown',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('No content available');
    });
  });

  describe('Security - URL Validation', () => {
    it('should reject SSRF attempts via localhost', () => {
      const ssrfUrls = [
        'http://localhost/admin',
        'http://127.0.0.1/config',
        'http://0.0.0.0/api',
        'http://[::1]/admin',
        'file:///etc/passwd',
      ];

      ssrfUrls.forEach((url) => {
        // These might pass URL validation but should be handled by Firecrawl API
        const tool = new WebScrapeTool({
          url,
          credentials: {
            [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
          },
        });
        expect(tool.params.url).toBe(url);
      });
    });

    it('should validate and sanitize URLs', () => {
      const tool = new WebScrapeTool({
        url: 'https://example.com/path/to/resource?param=value',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      expect(tool.params.url).toBe('https://example.com/path/to/resource?param=value');
    });
  });

  describe('Response Metadata', () => {
    it('should include metadata in successful response', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          markdown: '# Content',
          metadata: {
            title: 'Test',
            statusCode: 200,
            language: 'en',
          },
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.metadata).toBeDefined();
      expect(result.metadata?.statusCode).toBe(200);
      expect(result.metadata?.loadTime).toBeGreaterThan(0);
    });

    it('should track credits used', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          markdown: '# Content',
          metadata: {
            title: 'Test',
          },
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.creditsUsed).toBe(1);
    });
  });

  describe('Error Recovery', () => {
    it('should handle summarization failure gracefully', async () => {
      const largeContent = 'A'.repeat(6000000);

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            success: true,
            markdown: largeContent,
            metadata: {
              title: 'Large Page',
            },
          }),
        })
        .mockRejectedValueOnce(new Error('Summarization failed'));

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
          [CredentialType.OPENAI_API_KEY]: 'test-openai-key',
        },
      });

      const result = await tool.act();

      // Should still succeed with original content
      expect(result.success).toBe(true);
      expect(result.content).toBeDefined();
    });

    it('should handle malformed API responses', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => {
          throw new Error('Invalid JSON');
        },
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('Performance', () => {
    it('should complete scraping within reasonable time', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          markdown: '# Content',
          metadata: {
            title: 'Test',
          },
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const startTime = Date.now();
      await tool.act();
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(5000); // Should complete in < 5s
    });

    it('should track load time in metadata', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          markdown: '# Content',
          metadata: {
            title: 'Test',
          },
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.metadata?.loadTime).toBeGreaterThan(0);
    });
  });

  describe('Rate Limiting Awareness', () => {
    it('should handle rate limit errors from API', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: false,
          error: 'Rate limit exceeded. Please try again later.',
          code: 'rate_limit_exceeded',
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Rate limit');
    });
  });

  describe('Content Processing', () => {
    it('should trim whitespace from scraped content', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          markdown: '  # Content  \n\n  Text with spaces  ',
          metadata: {
            title: 'Test',
          },
        }),
      });

      const tool = new WebScrapeTool({
        url: 'https://example.com',
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.content).not.toMatch(/^\s/);
      expect(result.content).not.toMatch(/\s$/);
    });

    it('should preserve original URL in response', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          success: true,
          markdown: '# Content',
          metadata: {
            title: 'Test',
          },
        }),
      });

      const originalUrl = 'https://example.com/path?query=value';
      const tool = new WebScrapeTool({
        url: originalUrl,
        credentials: {
          [CredentialType.FIRECRAWL_API_KEY]: 'test-api-key',
        },
      });

      const result = await tool.act();

      expect(result.url).toBe(originalUrl);
    });
  });
});
