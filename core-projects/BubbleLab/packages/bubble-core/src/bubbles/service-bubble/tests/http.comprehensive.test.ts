/**
 * Comprehensive HTTP Bubble Tests
 * Unit, Security, and Resilience tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { HttpBubble } from '../http.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import {
  createMockResponse,
  createMockErrorResponse,
  securityPayloads,
  measurePerformance,
  createTestContext,
} from '../../tests/test-utils.js';

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('HttpBubble - Comprehensive Tests', () => {
  let testContext: ReturnType<typeof createTestContext>;

  beforeEach(() => {
    testContext = createTestContext();
    mockFetch.mockClear();
  });

  describe('Unit Tests - Validation', () => {
    it('should validate required inputs', () => {
      expect(() => {
        new HttpBubble({
          url: 'https://api.example.com/data',
          method: 'GET',
        });
      }).not.toThrow();
    });

    it('should reject invalid URL', () => {
      expect(() => {
        new HttpBubble({
          url: 'not-a-url',
          method: 'GET',
        });
      }).toThrow();
    });

    it('should reject invalid HTTP method', () => {
      expect(() => {
        new HttpBubble({
          url: 'https://api.example.com/data',
          method: 'INVALID' as any,
        });
      }).toThrow();
    });

    it('should validate body size limits', () => {
      const largeBody = 'x'.repeat(11 * 1024 * 1024); // 11MB

      expect(() => {
        new HttpBubble({
          url: 'https://api.example.com/data',
          method: 'POST',
          body: largeBody,
        });
      }).toThrow(/Request body exceeds maximum size/);
    });

    it('should validate timeout range', () => {
      expect(() => {
        new HttpBubble({
          url: 'https://api.example.com/data',
          timeout: 500, // Below minimum
        });
      }).toThrow();

      expect(() => {
        new HttpBubble({
          url: 'https://api.example.com/data',
          timeout: 150000, // Above maximum
        });
      }).toThrow();
    });

    it('should accept valid timeout range', () => {
      expect(() => {
        new HttpBubble({
          url: 'https://api.example.com/data',
          timeout: 30000, // Valid
        });
      }).not.toThrow();
    });
  });

  describe('Unit Tests - Operation', () => {
    it('should have correct static metadata', () => {
      expect(HttpBubble.bubbleName).toBe('http');
      expect(HttpBubble.service).toBe('nodex-core');
      expect(HttpBubble.type).toBe('service');
      expect(HttpBubble.alias).toBe('fetch');
      expect(HttpBubble.schema).toBeDefined();
      expect(HttpBubble.resultSchema).toBeDefined();
    });

    it('should make successful GET request', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'GET',
      });

      const result = await bubble.performAction(testContext);

      expect(result.success).toBe(true);
      expect(result.status).toBe(200);
      expect(result.json).toEqual({ success: true });
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should make POST request with JSON body', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ id: 123 }, 201));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/create',
        method: 'POST',
        body: { name: 'test' },
      });

      const result = await bubble.performAction(testContext);

      expect(result.success).toBe(true);
      expect(result.status).toBe(201);
      expect(result.json).toEqual({ id: 123 });
    });

    it('should handle different HTTP methods', async () => {
      const methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'] as const;

      for (const method of methods) {
        mockFetch.mockResolvedValue(createMockResponse({ method }));

        const bubble = new HttpBubble({
          url: 'https://api.example.com/data',
          method,
        });

        const result = await bubble.performAction(testContext);
        expect(result.success).toBe(true);
      }
    });

    it('should handle custom headers', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'GET',
        headers: {
          'X-Custom-Header': 'custom-value',
          Authorization: 'Bearer token123',
        },
      });

      await bubble.performAction(testContext);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/data',
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-Custom-Header': 'custom-value',
            Authorization: 'Bearer token123',
          }),
        })
      );
    });

    it('should not include body for GET requests', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'GET',
        body: { shouldIgnore: 'this' },
      });

      await bubble.performAction(testContext);

      const fetchCall = mockFetch.mock.calls[0];
      expect(fetchCall[1]).not.toHaveProperty('body');
    });

    it('should not include body for HEAD requests', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'HEAD',
        body: { shouldIgnore: 'this' },
      });

      await bubble.performAction(testContext);

      const fetchCall = mockFetch.mock.calls[0];
      expect(fetchCall[1]).not.toHaveProperty('body');
    });
  });

  describe('Unit Tests - Error Handling', () => {
    it('should handle HTTP errors gracefully', async () => {
      mockFetch.mockResolvedValue(createMockErrorResponse('Not found', 404));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/notfound',
      });

      const result = await bubble.performAction(testContext);

      expect(result.success).toBe(false);
      expect(result.status).toBe(404);
      expect(result.error).toContain('404');
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
      });

      const result = await bubble.performAction(testContext);

      expect(result.success).toBe(false);
      expect(result.status).toBe(0);
      expect(result.error).toBe('Network error');
    });

    it('should handle timeout errors', async () => {
      // Simulate timeout by aborting request
      const abortError = new Error('Request timeout');
      abortError.name = 'AbortError';
      mockFetch.mockRejectedValue(abortError);

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        timeout: 1000,
      });

      const result = await bubble.performAction(testContext);

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });

    it('should handle non-JSON responses', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => '<html><body>HTML Response</body></html>',
        headers: new Map([['content-type', 'text/html']]),
      });

      const bubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await bubble.performAction(testContext);

      expect(result.success).toBe(true);
      expect(result.body).toContain('<html>');
      expect(result.json).toBeUndefined();
    });
  });

  describe('Security Tests - SSRF Prevention', () => {
    it('should block localhost URLs', () => {
      const localhostUrls = [
        'http://localhost:8080/admin',
        'http://127.0.0.1:22',
        'http://127.0.0.1',
        'http://0.0.0.0:8080',
        'http://[::1]:8080',
      ];

      localhostUrls.forEach((url) => {
        expect(() => {
          new HttpBubble({ url, method: 'GET' });
        }).toThrow(/URL contains forbidden protocol, internal IP address/);
      });
    });

    it('should block private IP ranges', () => {
      const privateUrls = [
        'http://10.0.0.1/admin',
        'http://172.16.0.1/admin',
        'http://192.168.1.1/admin',
        'http://169.254.169.254/latest', // AWS metadata
        'http://100.100.100.200/latest', // GCP metadata
      ];

      privateUrls.forEach((url) => {
        expect(() => {
          new HttpBubble({ url, method: 'GET' });
        }).toThrow(/URL contains forbidden protocol, internal IP address/);
      });
    });

    it('should block cloud metadata endpoints', () => {
      const metadataUrls = [
        'http://metadata.google.internal/computeMetadata/v1/',
        'http://instance-data/latest/meta-data/',
        'http://linklocal.amazonaws.com/latest',
      ];

      metadataUrls.forEach((url) => {
        expect(() => {
          new HttpBubble({ url, method: 'GET' });
        }).toThrow(/URL contains forbidden protocol, internal IP address/);
      });
    });

    it('should block internal hostnames', () => {
      const internalHosts = [
        'http://local/resource',
        'http://broadcasthost/resource',
        'http://ip6-localhost/resource',
      ];

      internalHosts.forEach((url) => {
        expect(() => {
          new HttpBubble({ url, method: 'GET' });
        }).toThrow(/URL contains forbidden protocol, internal IP address/);
      });
    });

    it('should block non-HTTP protocols', () => {
      const invalidProtocols = [
        'file:///etc/passwd',
        'ftp://ftp.example.com/file',
        'javascript:alert(1)',
        'data:text/html,<script>alert(1)</script>',
      ];

      invalidProtocols.forEach((url) => {
        expect(() => {
          new HttpBubble({ url, method: 'GET' });
        }).toThrow(/URL contains forbidden protocol/);
      });
    });

    it('should allow valid public URLs', () => {
      const validUrls = [
        'https://api.example.com/data',
        'https://github.com/user/repo',
        'https://www.googleapis.com/drive/v3/files',
      ];

      validUrls.forEach((url) => {
        expect(() => {
          new HttpBubble({ url, method: 'GET' });
        }).not.toThrow();
      });
    });
  });

  describe('Security Tests - Redirect Handling', () => {
    it('should not follow redirects by default', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 302,
        statusText: 'Found',
        text: async () => '',
        headers: new Map([['location', 'http://localhost:8080/admin']]),
      });

      const bubble = new HttpBubble({
        url: 'https://api.example.com/redirect',
        followRedirects: false, // Default
      });

      await bubble.performAction(testContext);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/redirect',
        expect.objectContaining({
          redirect: 'manual',
        })
      );
    });

    it('should follow redirects when enabled', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 302,
        statusText: 'Found',
        text: async () => '',
        headers: new Map([['location', 'https://api.example.com/data']]),
      });

      const bubble = new HttpBubble({
        url: 'https://api.example.com/redirect',
        followRedirects: true,
      });

      await bubble.performAction(testContext);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/redirect',
        expect.objectContaining({
          redirect: 'follow',
        })
      );
    });
  });

  describe('Security Tests - Authentication', () => {
    it('should handle Bearer authentication', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'GET',
        authType: 'bearer',
        credentials: {
          [CredentialType.CUSTOM_AUTH_KEY]: 'test-token',
        },
      });

      await bubble.performAction(testContext);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/data',
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
          }),
        })
      );
    });

    it('should handle Basic authentication', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'GET',
        authType: 'basic',
        credentials: {
          [CredentialType.CUSTOM_AUTH_KEY]: 'base64-encoded-creds',
        },
      });

      await bubble.performAction(testContext);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/data',
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Basic base64-encoded-creds',
          }),
        })
      );
    });

    it('should handle API key authentication', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'GET',
        authType: 'api-key',
        credentials: {
          [CredentialType.CUSTOM_AUTH_KEY]: 'my-api-key',
        },
      });

      await bubble.performAction(testContext);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/data',
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-API-Key': 'my-api-key',
          }),
        })
      );
    });

    it('should handle custom header authentication', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'GET',
        authType: 'custom',
        authHeader: 'X-Custom-Auth',
        credentials: {
          [CredentialType.CUSTOM_AUTH_KEY]: 'custom-token',
        },
      });

      await bubble.performAction(testContext);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/data',
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-Custom-Auth': 'custom-token',
          }),
        })
      );
    });

    it('should handle no authentication', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/public/data',
        method: 'GET',
        authType: 'none',
      });

      await bubble.performAction(testContext);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/public/data',
        expect.objectContaining({
          headers: expect.not.objectContaining({
            Authorization: expect.any(String),
          }),
        })
      );
    });
  });

  describe('Resilience Tests - Timeout Handling', () => {
    it('should timeout after configured duration', async () => {
      // Create a promise that never resolves
      mockFetch.mockImplementation(
        () =>
          new Promise((resolve) => {
            // Never resolve
          })
      );

      const bubble = new HttpBubble({
        url: 'https://api.example.com/slow',
        timeout: 100, // 100ms timeout
      });

      const result = await bubble.performAction(testContext);

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    }, 10000);

    it('should complete within timeout for fast requests', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/fast',
        timeout: 5000,
      });

      const result = await bubble.performAction(testContext);

      expect(result.success).toBe(true);
      expect(result.responseTime).toBeLessThan(5000);
    });
  });

  describe('Resilience Tests - Retry Logic', () => {
    it('should handle transient failures', async () => {
      // First call fails, second succeeds
      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/unreliable',
      });

      const result = await bubble.performAction(testContext);

      // HTTP bubble doesn't have built-in retry logic
      // This test verifies the behavior
      expect(result.success).toBe(false);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('Performance Tests', () => {
    it('should complete requests quickly', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ success: true }));

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        method: 'GET',
      });

      const duration = await measurePerformance(
        () => bubble.performAction(testContext),
        100 // Should complete in less than 100ms
      );

      expect(duration).toBeLessThan(100);
    });

    it('should measure response time accurately', async () => {
      mockFetch.mockImplementation(
        async () =>
          new Promise((resolve) => {
            setTimeout(
              () =>
                resolve({
                  ok: true,
                  status: 200,
                  statusText: 'OK',
                  text: async () => '{"success":true}',
                  headers: new Map([['content-type', 'application/json']]),
                }),
              50 // 50ms delay
            );
          })
      );

      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
      });

      const result = await bubble.performAction(testContext);

      expect(result.responseTime).toBeGreaterThanOrEqual(50);
      expect(result.responseTime).toBeLessThan(200);
    });
  });

  describe('Credential Tests', () => {
    it('should test credentials successfully', async () => {
      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
      });

      const isValid = await bubble.testCredential();
      expect(isValid).toBe(true);
    });

    it('should handle missing credentials', async () => {
      const bubble = new HttpBubble({
        url: 'https://api.example.com/data',
        authType: 'bearer',
        // No credentials provided
      });

      // Should not throw, just proceed without auth
      expect(bubble).toBeDefined();
    });
  });
});
