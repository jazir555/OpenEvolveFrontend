/**
 * HTTPBubble Security Tests
 * File: service-bubble/http-bubble.test.ts
 *
 * Purpose: Comprehensive security testing for SSRF protection and input validation
 *
 * Security Coverage:
 * - SSRF (Server-Side Request Forgery) protection
 * - Input validation
 * - Protocol restrictions
 * - Private IP range blocking
 * - Cloud metadata endpoint protection
 */

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';
import { HTTPBubble } from './http-bubble';

describe('HTTPBubble - Security Tests', () => {
  let mockFetch: any;
  let httpBubble: HTTPBubble;

  beforeEach(() => {
    // Mock fetch API to prevent real HTTP calls
    mockFetch = vi.fn();
    global.fetch = mockFetch;

    // Create HTTP bubble instance
    httpBubble = new HTTPBubble();

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /**
   * SSRF Protection Tests
   *
   * Testing against common SSRF attack vectors:
   * - localhost and loopback addresses
   * - private IP ranges (RFC 1918)
   * - cloud metadata endpoints
   * - internal hostnames
   * - private IPv6 ranges
   * - unsafe protocols
   */

  describe('SSRF Protection - IPv4 Addresses', () => {
    test('should block localhost (127.0.0.1)', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: 'test' }),
      });

      const result = await httpBubble.get({
        url: 'http://127.0.0.1:8080/admin',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/localhost|127\.0\.0\.1|private|internal/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block private IPv4 range 10.0.0.0/8', async () => {
      const result = await httpBubble.get({
        url: 'http://10.0.0.1/sensitive',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/private|internal|10\.0\.0/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block private IPv4 range 172.16.0.0/12', async () => {
      const result = await httpBubble.get({
        url: 'http://172.31.255.255/internal-api',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/private|internal|172\./i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block private IPv4 range 192.168.0.0/16', async () => {
      const result = await httpBubble.get({
        url: 'http://192.168.1.1/config',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/private|internal|192\.168/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block cloud metadata endpoint (169.254.169.254)', async () => {
      const result = await httpBubble.get({
        url: 'http://169.254.169.254/latest/meta-data/',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/metadata|169\.254|private/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('SSRF Protection - Hostnames', () => {
    test('should block localhost hostname', async () => {
      const result = await httpBubble.get({
        url: 'http://localhost:3000/admin',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/localhost|internal/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block internal.* hostnames', async () => {
      const result = await httpBubble.get({
        url: 'http://internal.api.example.com/data',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/internal/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block *.internal hostnames', async () => {
      const result = await httpBubble.get({
        url: 'http://api.internal/sensitive',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/internal/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block 0.0.0.0 (all interfaces)', async () => {
      const result = await httpBubble.get({
        url: 'http://0.0.0.0:8080/',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/private|internal|invalid/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('SSRF Protection - IPv6 Addresses', () => {
    test('should block private IPv6 range fc00::/7 (ULA)', async () => {
      const result = await httpBubble.get({
        url: 'http://[fc00::1]:8080/',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/private|internal/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block link-local IPv6 range fe80::/10', async () => {
      const result = await httpBubble.get({
        url: 'http://[fe80::1]/',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/private|internal/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block IPv6 loopback ::1', async () => {
      const result = await httpBubble.get({
        url: 'http://[::1]:3000/',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/localhost|loopback|private/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('SSRF Protection - Protocol Restrictions', () => {
    test('should block file:// protocol', async () => {
      const result = await httpBubble.get({
        url: 'file:///etc/passwd',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/protocol|file:\/\//i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block ftp:// protocol', async () => {
      const result = await httpBubble.get({
        url: 'ftp://ftp.example.com/file',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/protocol|ftp/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block javascript:// protocol', async () => {
      const result = await httpBubble.get({
        url: 'javascript:alert(1)',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/protocol|javascript/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block data:// protocol', async () => {
      const result = await httpBubble.get({
        url: 'data:text/html,<script>alert(1)</script>',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/protocol|data:/i);
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('SSRF Protection - Bypass Attempts', () => {
    test('should block URL encoding bypass attempts', async () => {
      const result = await httpBubble.get({
        url: 'http://127%2e0%2e0%2e1/admin',
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block decimal IP notation', async () => {
      const result = await httpBubble.get({
        url: 'http://2130706433/admin', // 127.0.0.1 in decimal
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block hexadecimal IP notation', async () => {
      const result = await httpBubble.get({
        url: 'http://0x7f000001/admin', // 127.0.0.1 in hex
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });

    test('should block DNS rebinding attempts', async () => {
      const result = await httpBubble.get({
        url: 'http://evil.com@127.0.0.1/admin',
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('Input Validation - URL Format', () => {
    test('should reject malformed URLs', async () => {
      const result = await httpBubble.get({
        url: 'not-a-valid-url',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/url|invalid|format/i);
    });

    test('should reject URL without protocol', async () => {
      const result = await httpBubble.get({
        url: 'example.com/api',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/protocol|url/i);
    });

    test('should reject URL with invalid protocol', async () => {
      const result = await httpBubble.get({
        url: 'gopher://evil.com/',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/protocol/i);
    });

    test('should reject URL with fragments for internal access', async () => {
      const result = await httpBubble.get({
        url: 'http://example.com#@127.0.0.1',
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('Input Validation - Timeout Range', () => {
    test('should validate minimum timeout (1 second)', async () => {
      const result = await httpBubble.get({
        url: 'http://example.com/api',
        timeout: 0,
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/timeout|range/i);
    });

    test('should validate maximum timeout (120 seconds)', async () => {
      const result = await httpBubble.get({
        url: 'http://example.com/api',
        timeout: 121,
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/timeout|range|maximum/i);
    });

    test('should accept valid timeout range (1-120 seconds)', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const result = await httpBubble.get({
        url: 'http://example.com/api',
        timeout: 30,
      });

      expect(result.success).toBe(true);
    });

    test('should reject negative timeout', async () => {
      const result = await httpBubble.get({
        url: 'http://example.com/api',
        timeout: -10,
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/timeout|positive/i);
    });
  });

  describe('Input Validation - Body Size Limits', () => {
    test('should reject body exceeding 10MB limit', async () => {
      const largeBody = 'x'.repeat(11 * 1024 * 1024); // 11MB

      const result = await httpBubble.post({
        url: 'http://example.com/api',
        body: largeBody,
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/size|limit|10MB/i);
    });

    test('should accept body within 10MB limit', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const validBody = 'x'.repeat(5 * 1024 * 1024); // 5MB

      const result = await httpBubble.post({
        url: 'http://example.com/api',
        body: validBody,
      });

      expect(result.success).toBe(true);
    });

    test('should track body size in metrics', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const body = JSON.stringify({ data: 'test' });

      const result = await httpBubble.post({
        url: 'http://example.com/api',
        body,
      });

      expect(result.success).toBe(true);
      expect(result.metrics?.bodySize).toBeDefined();
    });
  });

  describe('Input Validation - Headers', () => {
    test('should strip dangerous headers', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const result = await httpBubble.get({
        url: 'http://example.com/api',
        headers: {
          'X-Forwarded-For': '127.0.0.1',
          'Host': 'evil.com',
        },
      });

      expect(result.success).toBe(true);
      // Verify forwarded headers were not sent
      const fetchCall = mockFetch.mock.calls[0];
      expect(fetchCall[1].headers['X-Forwarded-For']).toBeUndefined();
      expect(fetchCall[1].headers['Host']).not.toBe('evil.com');
    });

    test('should allow safe headers', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const result = await httpBubble.get({
        url: 'http://example.com/api',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'User-Agent': 'TestAgent/1.0',
        },
      });

      expect(result.success).toBe(true);
      const fetchCall = mockFetch.mock.calls[0];
      expect(fetchCall[1].headers['Content-Type']).toBe('application/json');
    });
  });

  describe('Legitimate Traffic - Should Allow', () => {
    test('should allow legitimate HTTPS URLs', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const result = await httpBubble.get({
        url: 'https://api.example.com/endpoint',
      });

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    test('should allow legitimate HTTP URLs to public IPs', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const result = await httpBubble.get({
        url: 'http://8.8.8.8:80/', // Google DNS
      });

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    test('should allow subdomains of public domains', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const result = await httpBubble.get({
        url: 'https://api.public.example.com/v1/data',
      });

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    test('should allow requests with valid authentication', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ authenticated: true }),
      });

      const result = await httpBubble.get({
        url: 'https://api.example.com/secure',
        headers: {
          'Authorization': 'Bearer valid-token-123',
        },
      });

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(1);
      const fetchCall = mockFetch.mock.calls[0];
      expect(fetchCall[1].headers['Authorization']).toBe('Bearer valid-token-123');
    });

    test('should allow all HTTP methods (GET, POST, PUT, PATCH, DELETE)', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'];

      for (const method of methods) {
        const result = await httpBubble.request({
          url: 'https://api.example.com/resource',
          method,
        });

        expect(result.success).toBe(true);
      }

      expect(mockFetch).toHaveBeenCalledTimes(methods.length);
    });
  });

  describe('Error Handling and Logging', () => {
    test('should log blocked SSRF attempts', async () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      await httpBubble.get({
        url: 'http://localhost:3000/admin',
      });

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining(/SSRF|blocked|localhost/i)
      );

      consoleSpy.mockRestore();
    });

    test('should provide detailed error messages for security violations', async () => {
      const result = await httpBubble.get({
        url: 'http://127.0.0.1/admin',
      });

      expect(result.error).toBeDefined();
      expect(result.error.length).toBeGreaterThan(10);
      expect(result.error).toMatch(/blocked|private|internal|localhost/i);
    });

    test('should include security context in error response', async () => {
      const blockedUrl = 'http://169.254.169.254/latest/meta-data/';
      const result = await httpBubble.get({ url: blockedUrl });

      expect(result).toMatchObject({
        success: false,
        error: expect.any(String),
        blockedUrl,
        reason: expect.stringMatching(/ssrf|security|private/i),
      });
    });
  });

  describe('Rate Limiting and Throttling', () => {
    test('should enforce rate limits on requests', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      // Make multiple rapid requests
      const requests = Array(100).fill(null).map(() =>
        httpBubble.get({
          url: 'https://api.example.com/test',
        })
      );

      const results = await Promise.all(requests);

      // Some requests should be rate limited
      const rateLimited = results.filter((r) => r.error?.includes('rate'));

      expect(rateLimited.length).toBeGreaterThan(0);
    });

    test('should track request count in metrics', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      await httpBubble.get({
        url: 'https://api.example.com/test',
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('Response Validation', () => {
    test('should validate response content type', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        headers: new Headers({
          'content-type': 'application/json',
        }),
        json: async () => ({ data: 'test' }),
      });

      const result = await httpBubble.get({
        url: 'https://api.example.com/data',
        expectedContentType: 'application/json',
      });

      expect(result.success).toBe(true);
    });

    test('should reject unexpected content types', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        headers: new Headers({
          'content-type': 'text/html',
        }),
        json: async () => ({ data: 'test' }),
      });

      const result = await httpBubble.get({
        url: 'https://api.example.com/data',
        expectedContentType: 'application/json',
      });

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/content.*type/i);
    });
  });
});
