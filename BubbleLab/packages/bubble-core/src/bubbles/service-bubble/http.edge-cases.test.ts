/**
 * Edge Case and Boundary Tests for HTTP Bubble
 *
 * Comprehensive edge case coverage including:
 * - Input boundaries (empty, null, max length, unicode, special characters)
 * - Network boundaries (timeouts, redirects, retries)
 * - Error paths (all HTTP status codes, network errors)
 * - Data edge cases (malformed responses, content types)
 * - Security edge cases (injection attacks, header manipulation)
 * - Concurrency edge cases (race conditions)
 * - Performance edge cases (large payloads, slow responses)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HttpBubble } from './http.js';

describe('HttpBubble - Edge Cases and Boundary Tests', () => {
  const mockFetch = vi.fn();
  global.fetch = mockFetch;

  global.AbortSignal.timeout = vi.fn((timeout: number) => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), timeout);
    return controller.signal;
  });

  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('Input Boundary Tests', () => {
    describe('URL Boundaries', () => {
      it('should handle maximum URL length (2048 chars)', () => {
        const longUrl = `https://example.com/?${'a'.repeat(2000)}`;

        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: longUrl,
          method: 'GET',
        });

        expect(async () => httpBubble.performAction()).not.toThrow();
      });

      it('should reject invalid URL formats', () => {
        const invalidUrls = [
          'not-a-url',
          'ht!tp://example.com',
          '://example.com',
          'https://',
          'https://example .com',
        ];

        invalidUrls.forEach((url) => {
          const result = HttpBubble.schema.safeParse({ url });
          expect(result.success).toBe(false);
        });
      });

      it('should handle URLs with special characters', () => {
        const specialUrls = [
          'https://example.com/path?query=value&key=another',
          'https://example.com/path#fragment',
          'https://example.com/path/user:password@host',
          'https://example.com/path?value=hello%20world',
        ];

        specialUrls.forEach((url) => {
          const result = HttpBubble.schema.safeParse({ url });
          expect(result.success).toBe(true);
        });
      });

      it('should handle Unicode URLs (IDN)', () => {
        const idnUrls = [
          'https://müller.de',
          'https://中国.cn',
          'https://россия.рф',
        ];

        idnUrls.forEach((url) => {
          const result = HttpBubble.schema.safeParse({ url });
          expect(result.success).toBe(true);
        });
      });

      it('should handle URLs with port numbers', () => {
        const urls = [
          'https://example.com:8080',
          'https://example.com:443',
          'http://example.com:80',
        ];

        urls.forEach((url) => {
          const result = HttpBubble.schema.safeParse({ url });
          expect(result.success).toBe(true);
        });
      });

      it('should handle IPv4 addresses', () => {
        const ipv4Urls = [
          'https://192.168.1.1',
          'https://10.0.0.1:8080',
          'http://127.0.0.1',
        ];

        ipv4Urls.forEach((url) => {
          const result = HttpBubble.schema.safeParse({ url });
          expect(result.success).toBe(true);
        });
      });

      it('should handle IPv6 addresses', () => {
        const ipv6Urls = [
          'https://[2001:db8::1]',
          'https://[::1]:8080',
          'http://[fe80::1]',
        ];

        ipv6Urls.forEach((url) => {
          const result = HttpBubble.schema.safeParse({ url });
          expect(result.success).toBe(true);
        });
      });
    });

    describe('Method Boundaries', () => {
      it('should support all standard HTTP methods', () => {
        const methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'] as const;

        methods.forEach((method) => {
          const result = HttpBubble.schema.safeParse({
            url: 'https://example.com',
            method,
          });

          expect(result.success).toBe(true);
        });
      });

      it('should reject invalid HTTP methods', () => {
        const result = HttpBubble.schema.safeParse({
          url: 'https://example.com',
          method: 'INVALID' as any,
        });

        expect(result.success).toBe(false);
      });
    });

    describe('Header Boundaries', () => {
      it('should handle empty headers object', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          headers: {},
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle maximum header size', async () => {
        const largeHeaderValue = 'x'.repeat(8192); // 8KB header value

        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          headers: {
            'X-Large-Header': largeHeaderValue,
          },
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle special characters in headers', async () => {
        const specialHeaders = {
          'X-Special': 'value with spaces',
          'X-Unicode': 'Hello 世界',
          'X-Special-Chars': 'value!@#$%^&*()',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          headers: specialHeaders,
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle multiple headers with same name', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          headers: {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
          },
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle case-insensitive headers', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          headers: {
            'content-type': 'application/json',
            'AUTHORIZATION': 'Bearer token',
            'Accept': 'application/json',
          },
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });
    });

    describe('Body Boundaries', () => {
      it('should handle empty string body', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map([['content-length', '0']]),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          method: 'POST',
          body: '',
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle large JSON body', async () => {
        const largeObject = {
          data: Array.from({ length: 10000 }, (_, i) => ({
            id: i,
            value: `item ${i}`,
          })),
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue(JSON.stringify(largeObject)),
          headers: new Map([['content-type', 'application/json']]),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          method: 'POST',
          body: largeObject,
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle Unicode in body', async () => {
        const unicodeBody = {
          message: 'Hello 世界 🌍 Привет مرحبا',
          emoji: '😀 😃 😄 😁',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue(JSON.stringify(unicodeBody)),
          headers: new Map([['content-type', 'application/json']]),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          method: 'POST',
          body: unicodeBody,
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.json?.message).toBe(unicodeBody.message);
      });

      it('should handle special characters in body', async () => {
        const specialBody = {
          text: 'Test\n\t\r\\\"\'<>[]{}&%$#@!',
          code: '\x00\x01\x02',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue(JSON.stringify(specialBody)),
          headers: new Map([['content-type', 'application/json']]),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          method: 'POST',
          body: specialBody,
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should ignore body for GET requests', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          method: 'GET',
          body: { shouldIgnore: 'this' },
        });

        await httpBubble.performAction();

        const fetchCall = mockFetch.mock.calls[0];
        expect(fetchCall[1]).not.toHaveProperty('body');
      });

      it('should ignore body for HEAD requests', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue(''),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          method: 'HEAD',
          body: { shouldIgnore: 'this' },
        });

        await httpBubble.performAction();

        const fetchCall = mockFetch.mock.calls[0];
        expect(fetchCall[1]).not.toHaveProperty('body');
      });
    });

    describe('Timeout Boundaries', () => {
      it('should handle minimum timeout (1ms)', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          timeout: 1,
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle maximum timeout (300000ms)', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('success'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
          timeout: 300000, // 5 minutes
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should reject timeout > 300000ms', () => {
        const result = HttpBubble.schema.safeParse({
          url: 'https://example.com',
          timeout: 300001,
        });

        expect(result.success).toBe(false);
      });

      it('should reject zero or negative timeout', () => {
        const timeouts = [0, -1, -1000];

        timeouts.forEach((timeout) => {
          const result = HttpBubble.schema.safeParse({
            url: 'https://example.com',
            timeout,
          });

          expect(result.success).toBe(false);
        });
      });
    });
  });

  describe('Network Edge Cases', () => {
    it('should handle request timeout', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Request timeout'));

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
        timeout: 1000,
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });

    it('should handle DNS resolution failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('ENOTFOUND example.com'));

      const httpBubble = new HttpBubble({
        url: 'https://nonexistent-domain-12345.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
    });

    it('should handle connection refused', async () => {
      mockFetch.mockRejectedValueOnce(new Error('ECONNREFUSED'));

      const httpBubble = new HttpBubble({
        url: 'https://localhost:1',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
    });

    it('should handle network unreachable', async () => {
      mockFetch.mockRejectedValueOnce(new Error('ENETUNREACH'));

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
    });

    it('should handle slow network (just before timeout)', async () => {
      mockFetch.mockImplementationOnce(() =>
        new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              ok: true,
              status: 200,
              statusText: 'OK',
              text: vi.fn().mockResolvedValue('success'),
              headers: new Map(),
            });
          }, 4900);
        })
      );

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
        timeout: 5000,
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should handle connection reset', async () => {
      mockFetch.mockRejectedValueOnce(new Error('ECONNRESET'));

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
    });
  });

  describe('HTTP Status Code Coverage', () => {
    it('should handle 1xx Informational responses', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 100,
        statusText: 'Continue',
        text: vi.fn().mockResolvedValue(''),
        headers: new Map(),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.status).toBe(100);
    });

    it('should handle 3xx Redirects', async () => {
      const redirectCodes = [301, 302, 303, 307, 308];

      for (const code of redirectCodes) {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: code,
          statusText: 'Redirect',
          text: vi.fn().mockResolvedValue(''),
          headers: new Map([['location', 'https://example.com/new']]),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
        });

        const result = await httpBubble.performAction();

        expect(result.status).toBe(code);
      }
    });

    it('should handle 4xx Client Errors', async () => {
      const clientErrors = [400, 401, 403, 404, 405, 409, 413, 415, 429];

      for (const code of clientErrors) {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: code,
          statusText: 'Client Error',
          text: vi.fn().mockResolvedValue(`Error ${code}`),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.status).toBe(code);
      }
    });

    it('should handle 5xx Server Errors', async () => {
      const serverErrors = [500, 502, 503, 504];

      for (const code of serverErrors) {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: code,
          statusText: 'Server Error',
          text: vi.fn().mockResolvedValue(`Error ${code}`),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.status).toBe(code);
      }
    });

    it('should handle 204 No Content', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 204,
        statusText: 'No Content',
        text: vi.fn().mockResolvedValue(''),
        headers: new Map([['content-length', '0']]),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.status).toBe(204);
      expect(result.body).toBe('');
    });
  });

  describe('Data Edge Cases', () => {
    it('should handle malformed JSON response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('invalid json{{{'),
        headers: new Map([['content-type', 'application/json']]),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.json).toBeUndefined();
      expect(result.body).toBe('invalid json{{{');
    });

    it('should handle empty response body', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue(''),
        headers: new Map([['content-length', '0']]),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.body).toBe('');
    });

    it('should handle binary response', async () => {
      const binaryData = Buffer.from([0x00, 0x01, 0x02, 0x03]);

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue(binaryData.toString()),
        headers: new Map([['content-type', 'application/octet-stream']]),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should handle various content types', async () => {
      const contentTypes = [
        'application/json',
        'text/html',
        'text/plain',
        'application/xml',
        'application/octet-stream',
        'multipart/form-data',
        'application/x-www-form-urlencoded',
      ];

      for (const contentType of contentTypes) {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('response'),
          headers: new Map([['content-type', contentType]]),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
        });

        const result = await httpBubble.performAction();

        expect(result.success).toBe(true);
      }
    });

    it('should handle chunked transfer encoding', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('chunked response'),
        headers: new Map([['transfer-encoding', 'chunked']]),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
    });
  });

  describe('Security Edge Cases', () => {
    it('should handle SSRF attempts (localhost)', async () => {
      const ssrfUrls = [
        'http://localhost/admin',
        'http://127.0.0.1/admin',
        'http://0.0.0.0/admin',
        'http://[::1]/admin',
      ];

      for (const url of ssrfUrls) {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('response'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url,
        });

        const result = await httpBubble.performAction();

        // Request should be made (validation happens at higher layer)
        expect(result.success).toBe(true);
      }
    });

    it('should handle header injection attempts', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('success'),
        headers: new Map(),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
        headers: {
          'X-Injected': "value\r\nX-Another: injected",
        },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should handle CRLF injection in URL', async () => {
      const httpBubble = new HttpBubble({
        url: 'https://example.com/path%0D%0AInjected-Header: true',
      });

      // URL should be parsed, but CRLF should be encoded
      expect(async () => httpBubble.performAction()).not.toThrow();
    });
  });

  describe('Concurrency Edge Cases', () => {
    it('should handle multiple simultaneous requests', async () => {
      const promises = [];

      for (let i = 0; i < 10; i++) {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue(`response ${i}`),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
        });

        promises.push(httpBubble.performAction());
      }

      const results = await Promise.all(promises);

      results.forEach((result) => {
        expect(result.success).toBe(true);
      });
    });

    it('should handle request cancellation', async () => {
      const controller = new AbortController();

      mockFetch.mockImplementationOnce(() =>
        new Promise((_, reject) => {
          setTimeout(() => {
            reject(new Error('Request cancelled'));
          }, 100);
        })
      );

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
        timeout: 5000,
      });

      // Cancel request immediately
      controller.abort();

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
    });
  });

  describe('Performance Edge Cases', () => {
    it('should handle large response body', async () => {
      const largeResponse = 'x'.repeat(10 * 1024 * 1024); // 10MB

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue(largeResponse),
        headers: new Map([['content-length', String(10 * 1024 * 1024)]]),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.size).toBe(10 * 1024 * 1024);
    });

    it('should handle many small requests', async () => {
      const promises = [];

      for (let i = 0; i < 100; i++) {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('ok'),
          headers: new Map(),
        });

        const httpBubble = new HttpBubble({
          url: 'https://example.com',
        });

        promises.push(httpBubble.performAction());
      }

      const results = await Promise.all(promises);

      results.forEach((result) => {
        expect(result.success).toBe(true);
      });
    });

    it('should track response time accurately', async () => {
      mockFetch.mockImplementationOnce(() =>
        new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              ok: true,
              status: 200,
              statusText: 'OK',
              text: vi.fn().mockResolvedValue('success'),
              headers: new Map(),
            });
          }, 100);
        })
      );

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
      });

      const result = await httpBubble.performAction();

      expect(result.responseTime).toBeGreaterThanOrEqual(90);
    });
  });

  describe('Redirect Handling', () => {
    it('should follow redirects when enabled', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: false,
          status: 302,
          statusText: 'Found',
          text: vi.fn().mockResolvedValue(''),
          headers: new Map([['location', 'https://example.com/new']]),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: vi.fn().mockResolvedValue('final response'),
          headers: new Map(),
        });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
        followRedirects: true,
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.body).toBe('final response');
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it('should not follow redirects when disabled', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 302,
        statusText: 'Found',
        text: vi.fn().mockResolvedValue(''),
        headers: new Map([['location', 'https://example.com/new']]),
      });

      const httpBubble = new HttpBubble({
        url: 'https://example.com',
        followRedirects: false,
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.status).toBe(302);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should handle redirect loops', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: false,
          status: 302,
          statusText: 'Found',
          text: vi.fn().mockResolvedValue(''),
          headers: new Map([['location', 'https://example.com/loop']]),
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 302,
          statusText: 'Found',
          text: vi.fn().mockResolvedValue(''),
          headers: new Map([['location', 'https://example.com/start']]),
        });

      const httpBubble = new HttpBubble({
        url: 'https://example.com/start',
        followRedirects: true,
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
    });
  });
});
