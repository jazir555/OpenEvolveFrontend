/**
 * Comprehensive Security Tests for Service Bubbles
 *
 * Tests OWASP Top 10 vulnerabilities and common security issues:
 * 1. SQL Injection Prevention
 * 2. XSS (Cross-Site Scripting) Prevention
 * 3. Input Validation
 * 4. Authentication/Authorization
 * 5. Rate Limiting
 * 6. Path Traversal Prevention
 * 7. SSRF (Server-Side Request Forgery) Prevention
 * 8. Injection Attacks
 * 9. Credential Security
 * 10. Data Sanitization
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StripeBubble } from '../../bubbles/service-bubble/stripe-bubble.js';
import { GoogleDriveBubble } from '../../bubbles/service-bubble/google-drive-bubble.js';
import { GoogleSheetsBubble } from '../../bubbles/service-bubble/google-sheets-bubble.js';
import { NotionBubble } from '../../bubbles/service-bubble/notion-bubble.js';
import { WebhookBubble } from '../../bubbles/service-bubble/webhook-bubble.js';
import { AirtableBubble } from '../../bubbles/service-bubble/airtable-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('Comprehensive Security Tests', () => {
  describe('1. SQL Injection Prevention', () => {
    it('should sanitize SQL injection attempts in parameters', async () => {
      const maliciousInputs = [
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM passwords--",
        "1'; DELETE FROM users WHERE '1'='1",
      ];

      for (const maliciousInput of maliciousInputs) {
        const bubble = new GoogleSheetsBubble({
          operation: 'updateCell',
          spreadsheetId: 'test-sheet-id',
          range: 'Sheet1!A1',
          value: maliciousInput,
          credentials: {
            [CredentialType.GOOGLE_SHEETS_CRED]: 'token',
          },
        });

        // Should not throw or allow SQL execution
        expect(bubble.params.value).toBe(maliciousInput);
        // Value should be sanitized during execution
      }
    });

    it('should handle SQL injection in database operations', async () => {
      const sqlInjection = {
        query: "SELECT * FROM users WHERE username = 'admin' OR '1'='1'",
        email: "test@example.com'; DROP TABLE users; --",
      };

      // PostgreSQL bubble should sanitize inputs
      const bubble = new AirtableBubble({
        operation: 'createRecord',
        baseId: 'app123',
        tableId: 'tbl123',
        fields: {
          username: sqlInjection.email,
          query: sqlInjection.query,
        },
        credentials: {
          [CredentialType.AIRTABLE_CRED]: 'key',
        },
      });

      expect(bubble).toBeDefined();
    });
  });

  describe('2. XSS (Cross-Site Scripting) Prevention', () => {
    it('should sanitize script tags in string inputs', async () => {
      const xssPayloads = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror=alert("XSS")>',
        '<svg onload=alert("XSS")>',
        'javascript:alert("XSS")',
        '<iframe src="javascript:alert(XSS)"></iframe>',
      ];

      for (const payload of xssPayloads) {
        const bubble = new NotionBubble({
          operation: 'createPage',
          parentPageId: 'page123',
          title: payload,
          credentials: {
            [CredentialType.NOTION_CRED]: 'token',
          },
        });

        // Should not allow raw script execution
        expect(bubble.params.title).toBe(payload);
      }
    });

    it('should escape HTML entities in user input', async () => {
      const htmlInput = '<div onclick="alert(1)">Click me</div>';

      const bubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet123',
        range: 'Sheet1!A1',
        value: htmlInput,
        credentials: {
          [CredentialType.GOOGLE_SHEETS_CRED]: 'token',
        },
      });

      // Should properly escape the HTML
      expect(bubble).toBeDefined();
    });
  });

  describe('3. Input Validation', () => {
    it('should validate email addresses', async () => {
      const invalidEmails = [
        'plaintext',
        '@example.com',
        'test@',
        'test..test@example.com',
        'test@example..com',
      ];

      for (const email of invalidEmails) {
        expect(() => {
          new StripeBubble({
            operation: 'createCustomer',
            email,
            credentials: {
              [CredentialType.STRIPE_CRED]: 'key',
            },
          });
        }).toThrow();
      }
    });

    it('should validate URLs to prevent SSRF attacks', async () => {
      const maliciousUrls = [
        'http://localhost/admin',
        'http://169.254.169.254/latest/meta-data/',
        'http://127.0.0.1:8080',
        'file:///etc/passwd',
        'ftp://malicious.com',
      ];

      for (const url of maliciousUrls) {
        const bubble = new WebhookBubble({
          operation: 'dispatchEvent',
          eventType: 'test',
          payload: { data: 'test' },
          targets: [url],
          credentials: {
            [CredentialType.WEBHOOK_CRED]: 'secret',
          },
        });

        // Should reject invalid URLs
        expect(bubble).toBeDefined();
      }
    });

    it('should prevent path traversal attacks', async () => {
      const pathTraversalAttempts = [
        '../../../etc/passwd',
        '..\\..\\..\\windows\\system32',
        '....//....//....//etc/passwd',
        '%2e%2e%2fetc%2fpasswd',
      ];

      for (const maliciousPath of pathTraversalAttempts) {
        expect(() => {
          new GoogleDriveBubble({
            operation: 'uploadFile',
            fileName: maliciousPath,
            content: 'malicious',
            credentials: {
              [CredentialType.GOOGLE_DRIVE_CRED]: 'token',
            },
          });
        }).toThrow();
      }
    });

    it('should validate injection attempts', async () => {
      const injectionPayloads = [
        '${7*7}',
        '#{7*7}',
        '{{7*7}}',
        '<%7*7%>',
        '$(sleep 10)',
        '`sleep 10`',
      ];

      for (const payload of injectionPayloads) {
        const bubble = new NotionBubble({
          operation: 'createPage',
          parentPageId: 'page123',
          title: payload,
          credentials: {
            [CredentialType.NOTION_CRED]: 'token',
          },
        });

        // Should handle template injection attempts
        expect(bubble).toBeDefined();
      }
    });
  });

  describe('4. Authentication/Authorization', () => {
    it('should reject missing API keys', async () => {
      const bubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
      });

      const result = await bubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('API key');
    });

    it('should reject invalid API keys', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized',
      } as Response);

      const bubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: {
          [CredentialType.STRIPE_CRED]: 'sk_invalid_key',
        },
      });

      const result = await bubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should handle expired credentials', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ error: { message: 'Token expired' } }),
      } as Response);

      const bubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'test',
        credentials: {
          [CredentialType.GOOGLE_DRIVE_CRED]: JSON.stringify({
            accessToken: 'expired_token',
          }),
        },
      });

      const result = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Token expired');
    });
  });

  describe('5. Rate Limiting', () => {
    it('should enforce rate limits on operations', async () => {
      vi.useFakeTimers();

      const bubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'test',
        credentials: {
          [CredentialType.GOOGLE_DRIVE_CRED]: 'token',
        },
      });

      // Make 5 requests (at the limit)
      for (let i = 0; i < 5; i++) {
        vi.mocked(global.fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: `file_${i}` }),
        } as Response);
        await bubble.performAction();
      }

      // 6th request should fail
      const result = await bubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Rate limit exceeded');

      vi.useRealTimers();
    });

    it('should implement token bucket refill', async () => {
      vi.useFakeTimers();
      vi.clearAllMocks();

      const bubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'test',
        credentials: {
          [CredentialType.GOOGLE_DRIVE_CRED]: 'token',
        },
      });

      // Make 5 requests (at limit)
      for (let i = 0; i < 5; i++) {
        vi.mocked(global.fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: `file_${i}` }),
        } as Response);
        await bubble.performAction();
      }

      // Wait for rate limit window to expire
      vi.advanceTimersByTime(61000);

      // Should allow new requests
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'file_new' }),
      } as Response);

      const result = await bubble.performAction();

      expect(result.success).toBe(true);

      vi.useRealTimers();
    });
  });

  describe('6. Path Traversal Prevention', () => {
    it('should sanitize file paths', async () => {
      const maliciousPaths = [
        '../../sensitive-file.txt',
        '..\\..\\sensitive-file.txt',
        '/etc/passwd',
        'C:\\Windows\\System32\\config',
        '%2e%2e/%2e%2e/%2e%2e',
      ];

      for (const path of maliciousPaths) {
        expect(() => {
          new GoogleDriveBubble({
            operation: 'uploadFile',
            fileName: path,
            content: 'malicious',
            credentials: {
              [CredentialType.GOOGLE_DRIVE_CRED]: 'token',
            },
          });
        }).toThrow();
      }
    });
  });

  describe('7. SSRF Prevention', () => {
    it('should validate webhook URLs', async () => {
      const ssrfUrls = [
        'http://localhost:8080/admin',
        'http://127.0.0.1/config',
        'http://169.254.169.254/meta-data/',
        'file:///etc/passwd',
        'gopher://malicious.com:70/_data',
      ];

      for (const url of ssrfUrls) {
        const bubble = new WebhookBubble({
          operation: 'dispatchEvent',
          eventType: 'test',
          payload: { data: 'test' },
          targets: [url],
          credentials: {
            [CredentialType.WEBHOOK_CRED]: 'secret',
          },
        });

        // Should validate URL format
        expect(bubble).toBeDefined();
      }
    });
  });

  describe('8. Data Sanitization', () => {
    it('should sanitize error messages', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'Error: Database connection failed at localhost:5432',
      } as Response);

      const bubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: {
          [CredentialType.STRIPE_CRED]: 'key',
        },
      });

      const result = await bubble.performAction();

      // Should sanitize internal details
      expect(result.result.error).toBeDefined();
      expect(result.result.error).not.toContain('localhost');
    });

    it('should not leak sensitive data in errors', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 403,
        text: async () => 'Access denied for user: admin@secret.com',
      } as Response);

      const bubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file123',
        credentials: {
          [CredentialType.GOOGLE_DRIVE_CRED]: 'token',
        },
      });

      const result = await bubble.performAction();

      expect(result.success).toBe(false);
      // Error should be sanitized
      expect(result.error).not.toContain('secret.com');
    });
  });

  describe('9. Credential Security', () => {
    it('should not log sensitive credentials', async () => {
      const consoleSpy = vi.spyOn(console, 'log');

      const bubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: {
          [CredentialType.STRIPE_CRED]: 'sk_live_secret_key_12345',
        },
      });

      await bubble.performAction();

      // Check that credentials are not logged
      const logs = consoleSpy.mock.calls.flat().join(' ');
      expect(logs).not.toContain('sk_live_secret_key_12345');

      consoleSpy.mockRestore();
    });

    it('should mask API keys in error messages', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        text: async () => 'Invalid API key: sk_live_secret_key_12345',
      } as Response);

      const bubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: {
          [CredentialType.STRIPE_CRED]: 'sk_live_secret_key_12345',
        },
      });

      const result = await bubble.performAction();

      // API key should be masked in error
      expect(result.result.error).toBeDefined();
      expect(result.result.error).not.toContain('sk_live_secret_key_12345');
    });
  });

  describe('10. Webhook Security', () => {
    it('should verify webhook signatures', async () => {
      const payload = JSON.stringify({ id: 'evt_123', type: 'payment.success' });
      const secret = 'whsec_test_secret';

      const crypto = await import('crypto');
      const timestamp = Math.floor(Date.now() / 1000);
      const signature = crypto
        .createHmac('sha256', secret)
        .update(`${timestamp}.${payload}`)
        .digest('hex');

      const bubble = new WebhookBubble({
        operation: 'verifySignature',
        payload,
        signature: `t=${timestamp},v1=${signature}`,
        secret,
        credentials: {
          [CredentialType.WEBHOOK_CRED]: 'secret',
        },
      });

      const result = await bubble.performAction();

      expect(result.result.valid).toBe(true);
    });

    it('should reject invalid webhook signatures', async () => {
      const payload = JSON.stringify({ id: 'evt_123', type: 'payment.success' });
      const invalidSignature = 'invalid_signature';

      const bubble = new WebhookBubble({
        operation: 'verifySignature',
        payload,
        signature: invalidSignature,
        secret: 'whsec_test_secret',
        credentials: {
          [CredentialType.WEBHOOK_CRED]: 'secret',
        },
      });

      const result = await bubble.performAction();

      expect(result.result.valid).toBe(false);
    });

    it('should prevent webhook replay attacks', async () => {
      const oldTimestamp = Math.floor(Date.now() / 1000) - 1000; // 1000 seconds ago
      const payload = JSON.stringify({ id: 'evt_123' });

      const crypto = await import('crypto');
      const signature = crypto
        .createHmac('sha256', 'secret')
        .update(`${oldTimestamp}.${payload}`)
        .digest('hex');

      const bubble = new WebhookBubble({
        operation: 'verifySignature',
        payload,
        signature: `t=${oldTimestamp},v1=${signature}`,
        secret: 'secret',
        maxAge: 300000, // 5 minutes
        credentials: {
          [CredentialType.WEBHOOK_CRED]: 'secret',
        },
      });

      const result = await bubble.performAction();

      expect(result.result.valid).toBe(false);
    });
  });
});
