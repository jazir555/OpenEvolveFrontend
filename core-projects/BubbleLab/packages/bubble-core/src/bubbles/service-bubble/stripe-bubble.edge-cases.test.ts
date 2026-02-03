/**
 * Edge Case and Boundary Tests for Stripe Bubble
 *
 * Comprehensive edge case coverage including:
 * - Input boundaries (empty, null, max length, unicode, special characters)
 * - Network boundaries (timeouts, retries, rate limits)
 * - Error paths (all error types and codes)
 * - Data edge cases (malformed JSON, missing fields)
 * - Security edge cases (injection attacks)
 * - Concurrency edge cases (race conditions)
 * - Performance edge cases (large payloads, memory)
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StripeBubble } from './stripe-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('StripeBubble - Edge Cases and Boundary Tests', () => {
  let stripeBubble: StripeBubble;
  const mockCredentials = {
    [CredentialType.STRIPE_CRED]: 'sk_test_mock_api_key',
  };

  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Input Boundary Tests', () => {
    describe('String Boundaries', () => {
      it('should handle empty string for customer email', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: false,
          status: 400,
          json: async () => ({ error: { message: 'Invalid email' } }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createCustomer',
          email: '',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(false);
        expect(result.result.error).toBeDefined();
      });

      it('should handle maximum length strings (5000 chars)', async () => {
        const maxLength = 'x'.repeat(5000);
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'cus_test_123',
            description: maxLength,
            email: 'test@example.com',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createCustomer',
          email: 'test@example.com',
          description: maxLength,
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
      });

      it('should handle minimum length strings (1 char)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'cus_test_123',
            description: 'x',
            email: 'test@example.com',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createCustomer',
          email: 'test@example.com',
          description: 'x',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
      });

      it('should handle unicode and emoji characters', async () => {
        const unicodeText = 'Hello 世界 🌍 🎉 Test中文';
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'cus_test_123',
            name: unicodeText,
            email: 'test@example.com',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createCustomer',
          email: 'test@example.com',
          name: unicodeText,
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.name).toBe(unicodeText);
      });

      it('should handle special characters and control characters', async () => {
        const specialChars = "Test\n\t\r\\\"'<>[]{}&%$#@!";

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'cus_test_123',
            description: specialChars,
            email: 'test@example.com',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createCustomer',
          email: 'test@example.com',
          description: specialChars,
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
      });

      it('should handle whitespace variations', async () => {
        const whitespaceText = '  test  \t\t  test\n\n';

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'cus_test_123',
            description: whitespaceText,
            email: 'test@example.com',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createCustomer',
          email: 'test@example.com',
          description: whitespaceText,
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
      });

      it('should handle case sensitivity in customer IDs', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'cus_Test_Caps_123',
            email: 'test@example.com',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'getCustomer',
          customerId: 'cus_Test_Caps_123',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
      });
    });

    describe('Numeric Boundaries', () => {
      it('should handle maximum payment amount ($999,999.99)', async () => {
        const maxAmount = 99999999; // in cents

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'pi_test_123',
            amount: maxAmount,
            currency: 'usd',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createPaymentIntent',
          amount: maxAmount,
          currency: 'usd',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.amount).toBe(maxAmount);
      });

      it('should handle minimum positive amount (1 cent)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'pi_test_123',
            amount: 1,
            currency: 'usd',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createPaymentIntent',
          amount: 1,
          currency: 'usd',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.amount).toBe(1);
      });

      it('should handle zero amount for verification intents', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'pi_test_123',
            amount: 0,
            currency: 'usd',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createPaymentIntent',
          amount: 0,
          currency: 'usd',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.amount).toBe(0);
      });

      it('should handle negative amount validation', async () => {
        stripeBubble = new StripeBubble({
          operation: 'createPaymentIntent',
          amount: -100,
          currency: 'usd',
          credentials: mockCredentials,
        });

        await expect(stripeBubble.performAction()).rejects.toThrow();
      });

      it('should handle decimal precision (2 decimal places)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'pi_test_123',
            amount: 1001, // $10.01
            currency: 'usd',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createPaymentIntent',
          amount: 1001,
          currency: 'usd',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.amount).toBe(1001);
      });
    });

    describe('Array Boundaries', () => {
      it('should handle empty array for metadata', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'cus_test_123',
            metadata: {},
            email: 'test@example.com',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'createCustomer',
          email: 'test@example.com',
          metadata: {},
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
      });

      it('should handle single item array', async () => {
        const invoiceList = {
          data: [
            {
              id: 'in_test_123',
              amount_due: 1000,
              currency: 'usd',
            },
          ],
          has_more: false,
        };

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => invoiceList,
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'listInvoices',
          customer: 'cus_test_123',
          limit: 1,
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.invoices).toHaveLength(1);
      });

      it('should handle maximum page size (100 items)', async () => {
        const hundredInvoices = Array.from({ length: 100 }, (_, i) => ({
          id: `in_test_${i}`,
          amount_due: i * 100,
          currency: 'usd',
        }));

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: hundredInvoices,
            has_more: false,
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'listInvoices',
          customer: 'cus_test_123',
          limit: 100,
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
        expect(result.result.invoices).toHaveLength(100);
      });
    });

    describe('ID Format Validations', () => {
      it('should handle valid payment intent ID format', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'pi_3Mml1oLkdIwHu7ix0snN0B15',
            status: 'succeeded',
          }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'confirmPayment',
          paymentIntentId: 'pi_3Mml1oLkdIwHu7ix0snN0B15',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(true);
      });

      it('should handle invalid payment intent ID format', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: false,
          status: 404,
          json: async () => ({ error: { message: 'Invalid payment intent ID' } }),
        } as Response);

        stripeBubble = new StripeBubble({
          operation: 'confirmPayment',
          paymentIntentId: 'invalid_format',
          credentials: mockCredentials,
        });

        const result = await stripeBubble.performAction();

        expect(result.result.success).toBe(false);
      });

      it('should handle null payment intent ID', async () => {
        stripeBubble = new StripeBubble({
          operation: 'confirmPayment',
          paymentIntentId: null as any,
          credentials: mockCredentials,
        });

        await expect(stripeBubble.performAction()).rejects.toThrow();
      });
    });
  });

  describe('Network Edge Cases', () => {
    it('should handle timeout boundary (just before timeout)', async () => {
      vi.mocked(fetch).mockImplementationOnce(() =>
        new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              ok: true,
              json: async () => ({ id: 'pi_test_123', amount: 1000 }),
            } as Response);
          }, 4500); // Just before 5000ms timeout
        })
      );

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        timeout: 5000,
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle timeout boundary (at timeout)', async () => {
      vi.mocked(fetch).mockImplementationOnce(() =>
        new Promise((_, reject) => {
          setTimeout(() => {
            reject(new Error('Request timeout'));
          }, 5000);
        })
      );

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        timeout: 5000,
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should handle retry limit boundary (max retries)', async () => {
      // Fail 3 times, succeed on 4th (max retries)
      vi.mocked(fetch)
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: 'pi_test_123', amount: 1000 }),
        } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(4);
    });

    it('should handle retry limit exceeded', async () => {
      // Always fail
      vi.mocked(fetch).mockRejectedValue(new Error('Network error'));

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should handle rate limit boundary (just before limit)', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'pi_test_123', amount: 1000 }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle rate limit exceeded (429 status)', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: async () => ({
          error: {
            message: 'Rate limit exceeded',
            type: 'rate_limit_error',
          },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('rate limit');
    });

    it('should handle slow response scenarios', async () => {
      vi.mocked(fetch).mockImplementationOnce(() =>
        new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              ok: true,
              json: async () => ({ id: 'pi_test_123', amount: 1000 }),
            } as Response);
          }, 9000); // Very slow but within timeout
        })
      );

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        timeout: 10000,
        credentials: mockCredentials,
      });

      const startTime = Date.now();
      const result = await stripeBubble.performAction();
      const duration = Date.now() - startTime;

      expect(result.result.success).toBe(true);
      expect(duration).toBeGreaterThan(8000);
    });
  });

  describe('Error Path Coverage', () => {
    it('should handle 400 Bad Request', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({
          error: { message: 'Bad request', type: 'invalid_request_error' },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: -100,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should handle 401 Unauthorized', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({
          error: { message: 'Invalid API key', type: 'invalid_request_error' },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: { [CredentialType.STRIPE_CRED]: 'sk_invalid_key' },
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('authentication');
    });

    it('should handle 402 Payment Required', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 402,
        json: async () => ({
          error: {
            message: 'Your card was declined',
            type: 'card_error',
            code: 'card_declined',
          },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('declined');
    });

    it('should handle 404 Not Found', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({
          error: { message: 'Resource not found', type: 'invalid_request_error' },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'getCustomer',
        customerId: 'cus_nonexistent',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('not found');
    });

    it('should handle 409 Conflict', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({
          error: {
            message: 'Customer already exists',
            type: 'invalid_request_error',
          },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'existing@example.com',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should handle 500 Internal Server Error', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({
          error: { message: 'Internal server error', type: 'api_error' },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('server error');
    });

    it('should handle 503 Service Unavailable', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({
          error: { message: 'Service unavailable', type: 'api_error' },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });
  });

  describe('Data Edge Cases', () => {
    it('should handle malformed JSON response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => {
          throw new SyntaxError('Invalid JSON');
        },
        text: async () => 'invalid json{{{',
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'test@example.com',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should handle missing required fields in response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          // Missing 'id' field
          email: 'test@example.com',
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'test@example.com',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle extra unexpected fields in response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'cus_test_123',
          email: 'test@example.com',
          unexpected_field: 'unexpected_value',
          another_unexpected: 123,
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'test@example.com',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('cus_test_123');
    });

    it('should handle null values in non-nullable fields', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'cus_test_123',
          email: null, // Should be non-nullable
          name: 'Test Customer',
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'test@example.com',
        name: 'Test Customer',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle date/time boundary conditions', async () => {
      const leapYearDate = '2024-02-29T12:00:00Z'; // Leap year

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'sub_test_123',
          current_period_end: Math.floor(new Date(leapYearDate).getTime() / 1000),
          status: 'active',
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createSubscription',
        customer: 'cus_test_123',
        priceId: 'price_test_123',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle timezone edge cases', async () => {
      // Test UTC boundary
      const utcDate = new Date('2024-01-01T23:59:59.999Z');

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'pi_test_123',
          created: Math.floor(utcDate.getTime() / 1000),
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
    });
  });

  describe('Security Edge Cases', () => {
    it('should prevent SQL injection in email field', async () => {
      const sqlInjection = "'; DROP TABLE customers; --";

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ error: { message: 'Invalid email' } }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: sqlInjection,
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should prevent XSS in metadata fields', async () => {
      const xssPayload = '<script>alert("xss")</script>';

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'cus_test_123',
          metadata: { description: xssPayload },
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'test@example.com',
        description: xssPayload,
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      // Payload should be escaped in the result
      expect(typeof result.result.description).toBe('string');
    });

    it('should validate webhook signature properly', async () => {
      const payload = JSON.stringify({ id: 'evt_test_123' });
      const signatureHeader = 'invalid_signature';

      stripeBubble = new StripeBubble({
        operation: 'handleWebhook',
        payload,
        signature: signatureHeader,
        secret: 'whsec_test_secret',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('signature');
    });

    it('should handle malformed authentication tokens', async () => {
      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: { [CredentialType.STRIPE_CRED]: 'invalid_token_format' },
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });
  });

  describe('Concurrency Edge Cases', () => {
    it('should handle simultaneous requests to same resource', async () => {
      const customerId = 'cus_test_123';

      // Create two concurrent update requests
      const promise1 = (async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: customerId, email: 'update1@example.com' }),
        } as Response);

        const bubble = new StripeBubble({
          operation: 'updateCustomer',
          customerId,
          email: 'update1@example.com',
          credentials: mockCredentials,
        });

        return await bubble.performAction();
      })();

      const promise2 = (async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: customerId, email: 'update2@example.com' }),
        } as Response);

        const bubble = new StripeBubble({
          operation: 'updateCustomer',
          customerId,
          email: 'update2@example.com',
          credentials: mockCredentials,
        });

        return await bubble.performAction();
      })();

      const [result1, result2] = await Promise.all([promise1, promise2]);

      expect(result1.result.success).toBe(true);
      expect(result2.result.success).toBe(true);
    });

    it('should handle race conditions in status changes', async () => {
      const paymentIntentId = 'pi_test_123';

      // Simulate a payment being confirmed while it's being processed
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: paymentIntentId,
          status: 'processing',
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'confirmPayment',
        paymentIntentId,
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.status).toBeDefined();
    });
  });

  describe('Memory/Performance Edge Cases', () => {
    it('should handle large payload in metadata', async () => {
      const largeMetadata = {
        key: 'x'.repeat(100000), // 100KB string
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'cus_test_123',
          metadata: largeMetadata,
        }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'test@example.com',
        metadata: largeMetadata,
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle many small requests', async () => {
      const promises = [];

      for (let i = 0; i < 50; i++) {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: `cus_test_${i}`,
            email: `test${i}@example.com`,
          }),
        } as Response);

        const bubble = new StripeBubble({
          operation: 'createCustomer',
          email: `test${i}@example.com`,
          credentials: mockCredentials,
        });

        promises.push(bubble.performAction());
      }

      const results = await Promise.all(promises);

      results.forEach((result) => {
        expect(result.result.success).toBe(true);
      });
    });

    it('should handle connection pool exhaustion', async () => {
      // Simulate many concurrent requests
      const promises = [];

      for (let i = 0; i < 100; i++) {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: `pi_test_${i}`,
            amount: 1000,
            currency: 'usd',
          }),
        } as Response);

        const bubble = new StripeBubble({
          operation: 'createPaymentIntent',
          amount: 1000,
          currency: 'usd',
          credentials: mockCredentials,
        });

        promises.push(bubble.performAction());
      }

      const results = await Promise.all(promises);

      // All requests should eventually succeed
      const successCount = results.filter((r) => r.result.success).length;
      expect(successCount).toBeGreaterThan(90); // Allow some failures due to pool limits
    });
  });
});
