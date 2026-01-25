/**
 * Comprehensive tests for Stripe Bubble
 *
 * Tests all 15 operations:
 * 1. createPaymentIntent
 * 2. confirmPayment
 * 3. refundPayment
 * 4. createCustomer
 * 5. getCustomer
 * 6. updateCustomer
 * 7. createSubscription
 * 8. cancelSubscription
 * 9. updateSubscription
 * 10. createInvoice
 * 11. getInvoice
 * 12. listInvoices
 * 13. createProduct
 * 14. createPrice
 * 15. handleWebhook
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StripeBubble } from './stripe-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('StripeBubble', () => {
  let stripeBubble: StripeBubble;
  const mockCredentials = {
    [CredentialType.STRIPE_CRED]: 'sk_test_mock_api_key',
  };

  beforeEach(() => {
    // Mock fetch globally
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Operation 1: createPaymentIntent', () => {
    it('should create a payment intent successfully', async () => {
      const mockResponse = {
        id: 'pi_test_123',
        amount: 1000,
        currency: 'usd',
        status: 'requires_payment_method',
        client_secret: 'pi_test_123_secret_abc',
        description: 'Test payment',
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        description: 'Test payment',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('pi_test_123');
      expect(result.result.amount).toBe(1000);
      expect(result.result.currency).toBe('usd');
      expect(result.result.clientSecret).toBe('pi_test_123_secret_abc');
    });

    it('should handle authentication errors', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized',
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('authentication');
    });

    it('should handle rate limiting with retry', async () => {
      vi.mocked(fetch)
        .mockRejectedValueOnce(new Error('Rate limit exceeded'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'pi_test_123',
            amount: 1000,
            currency: 'usd',
            status: 'requires_payment_method',
            created: Math.floor(Date.now() / 1000),
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
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2);
    });

    it('should validate amount is positive', async () => {
      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: -100,
        currency: 'usd',
        credentials: mockCredentials,
      });

      await expect(stripeBubble.performAction()).rejects.toThrow();
    });
  });

  describe('Operation 2: confirmPayment', () => {
    it('should confirm a payment intent successfully', async () => {
      const mockResponse = {
        id: 'pi_test_123',
        amount: 1000,
        currency: 'usd',
        status: 'succeeded',
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'confirmPayment',
        paymentIntentId: 'pi_test_123',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('pi_test_123');
      expect(result.result.status).toBe('succeeded');
    });

    it('should handle invalid payment intent ID', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        text: async () => 'Not Found',
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'confirmPayment',
        paymentIntentId: 'pi_invalid',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('Not Found');
    });
  });

  describe('Operation 3: refundPayment', () => {
    it('should create a refund successfully', async () => {
      const mockResponse = {
        id: 're_test_123',
        amount: 1000,
        currency: 'usd',
        status: 'succeeded',
        payment_intent: 'pi_test_123',
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'refundPayment',
        paymentIntentId: 'pi_test_123',
        amount: 1000,
        reason: 'requested_by_customer',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('re_test_123');
      expect(result.result.amount).toBe(1000);
      expect(result.result.status).toBe('succeeded');
    });

    it('should handle partial refunds', async () => {
      const mockResponse = {
        id: 're_test_123',
        amount: 500,
        currency: 'usd',
        status: 'succeeded',
        payment_intent: 'pi_test_123',
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'refundPayment',
        paymentIntentId: 'pi_test_123',
        amount: 500,
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.amount).toBe(500);
    });
  });

  describe('Operation 4: createCustomer', () => {
    it('should create a customer successfully', async () => {
      const mockResponse = {
        id: 'cus_test_123',
        email: 'test@example.com',
        name: 'Test Customer',
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'test@example.com',
        name: 'Test Customer',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('cus_test_123');
      expect(result.result.email).toBe('test@example.com');
      expect(result.result.name).toBe('Test Customer');
    });

    it('should validate email format', async () => {
      stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'invalid-email',
        credentials: mockCredentials,
      });

      await expect(stripeBubble.performAction()).rejects.toThrow();
    });
  });

  describe('Operation 5: getCustomer', () => {
    it('should retrieve customer successfully', async () => {
      const mockResponse = {
        id: 'cus_test_123',
        email: 'test@example.com',
        name: 'Test Customer',
        phone: '+1234567890',
        description: 'Test customer',
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'getCustomer',
        customerId: 'cus_test_123',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('cus_test_123');
      expect(result.result.email).toBe('test@example.com');
    });

    it('should handle non-existent customer', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        text: async () => 'Customer not found',
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'getCustomer',
        customerId: 'cus_nonexistent',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('Customer not found');
    });
  });

  describe('Operation 6: updateCustomer', () => {
    it('should update customer successfully', async () => {
      const mockResponse = {
        id: 'cus_test_123',
        email: 'updated@example.com',
        name: 'Updated Name',
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'updateCustomer',
        customerId: 'cus_test_123',
        email: 'updated@example.com',
        name: 'Updated Name',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.email).toBe('updated@example.com');
      expect(result.result.name).toBe('Updated Name');
    });
  });

  describe('Operation 7: createSubscription', () => {
    it('should create a subscription successfully', async () => {
      const mockResponse = {
        id: 'sub_test_123',
        customer: 'cus_test_123',
        status: 'active',
        current_period_start: Math.floor(Date.now() / 1000),
        current_period_end: Math.floor(Date.now() / 1000) + 2592000,
        cancel_at_period_end: false,
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createSubscription',
        customer: 'cus_test_123',
        priceId: 'price_test_123',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('sub_test_123');
      expect(result.result.status).toBe('active');
    });
  });

  describe('Operation 8: cancelSubscription', () => {
    it('should cancel a subscription successfully', async () => {
      const mockResponse = {
        id: 'sub_test_123',
        customer: 'cus_test_123',
        status: 'canceled',
        current_period_start: Math.floor(Date.now() / 1000),
        current_period_end: Math.floor(Date.now() / 1000) + 2592000,
        cancel_at_period_end: true,
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'cancelSubscription',
        subscriptionId: 'sub_test_123',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.status).toBe('canceled');
      expect(result.result.cancelAtPeriodEnd).toBe(true);
    });
  });

  describe('Operation 9: updateSubscription', () => {
    it('should update subscription successfully', async () => {
      const mockResponse = {
        id: 'sub_test_123',
        customer: 'cus_test_123',
        status: 'active',
        current_period_start: Math.floor(Date.now() / 1000),
        current_period_end: Math.floor(Date.now() / 1000) + 2592000,
        cancel_at_period_end: false,
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'updateSubscription',
        subscriptionId: 'sub_test_123',
        priceId: 'price_new_123',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('sub_test_123');
    });
  });

  describe('Operation 10: createInvoice', () => {
    it('should create an invoice successfully', async () => {
      const mockResponse = {
        id: 'in_test_123',
        number: 'INV-001',
        status: 'draft',
        amount_due: 1000,
        currency: 'usd',
        customer: 'cus_test_123',
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createInvoice',
        customer: 'cus_test_123',
        description: 'Test invoice',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('in_test_123');
      expect(result.result.amountDue).toBe(1000);
    });
  });

  describe('Operation 11: getInvoice', () => {
    it('should retrieve invoice successfully', async () => {
      const mockResponse = {
        id: 'in_test_123',
        number: 'INV-001',
        status: 'paid',
        amount_due: 1000,
        currency: 'usd',
        customer: 'cus_test_123',
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'getInvoice',
        invoiceId: 'in_test_123',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('in_test_123');
      expect(result.result.status).toBe('paid');
    });
  });

  describe('Operation 12: listInvoices', () => {
    it('should list invoices successfully', async () => {
      const mockResponse = {
        data: [
          {
            id: 'in_test_123',
            number: 'INV-001',
            status: 'paid',
            amount_due: 1000,
            currency: 'usd',
            customer: 'cus_test_123',
            created: Math.floor(Date.now() / 1000),
          },
          {
            id: 'in_test_456',
            number: 'INV-002',
            status: 'open',
            amount_due: 2000,
            currency: 'usd',
            customer: 'cus_test_123',
            created: Math.floor(Date.now() / 1000),
          },
        ],
        has_more: false,
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'listInvoices',
        customer: 'cus_test_123',
        limit: 10,
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.invoices).toHaveLength(2);
      expect(result.result.count).toBe(2);
      expect(result.result.hasMore).toBe(false);
    });

    it('should handle pagination', async () => {
      const mockResponse = {
        data: [
          {
            id: 'in_test_789',
            number: 'INV-003',
            status: 'paid',
            amount_due: 3000,
            currency: 'usd',
            customer: 'cus_test_123',
            created: Math.floor(Date.now() / 1000),
          },
        ],
        has_more: true,
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'listInvoices',
        customer: 'cus_test_123',
        limit: 10,
        startingAfter: 'in_test_456',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.invoices).toHaveLength(1);
      expect(result.result.hasMore).toBe(true);
    });
  });

  describe('Operation 13: createProduct', () => {
    it('should create a product successfully', async () => {
      const mockResponse = {
        id: 'prod_test_123',
        name: 'Test Product',
        description: 'A test product',
        active: true,
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createProduct',
        name: 'Test Product',
        description: 'A test product',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('prod_test_123');
      expect(result.result.name).toBe('Test Product');
      expect(result.result.active).toBe(true);
    });
  });

  describe('Operation 14: createPrice', () => {
    it('should create a price successfully', async () => {
      const mockResponse = {
        id: 'price_test_123',
        product: 'prod_test_123',
        unit_amount: 1000,
        currency: 'usd',
        recurring: {
          interval: 'month',
          interval_count: 1,
        },
        active: true,
        created: Math.floor(Date.now() / 1000),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPrice',
        product: 'prod_test_123',
        unitAmount: 1000,
        currency: 'usd',
        recurring: {
          interval: 'month',
          intervalCount: 1,
        },
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('price_test_123');
      expect(result.result.unitAmount).toBe(1000);
      expect(result.result.recurring.interval).toBe('month');
    });
  });

  describe('Operation 15: handleWebhook', () => {
    it('should verify and handle webhook successfully', async () => {
      const payload = JSON.stringify({
        id: 'evt_test_123',
        type: 'payment_intent.succeeded',
        data: {
          object: {
            id: 'pi_test_123',
            amount: 1000,
          },
        },
      });

      const crypto = await import('crypto');
      const timestamp = Math.floor(Date.now() / 1000);
      const signedPayload = `${timestamp}.${payload}`;
      const signature = crypto
        .createHmac('sha256', 'whsec_test_secret')
        .update(signedPayload)
        .digest('hex');
      const signatureHeader = `t=${timestamp},v1=${signature}`;

      stripeBubble = new StripeBubble({
        operation: 'handleWebhook',
        payload,
        signature: signatureHeader,
        secret: 'whsec_test_secret',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.processed).toBe(true);
      expect(result.result.id).toBe('evt_test_123');
      expect(result.result.type).toBe('payment_intent.succeeded');
    });

    it('should reject invalid webhook signatures', async () => {
      const payload = JSON.stringify({
        id: 'evt_test_123',
        type: 'payment_intent.succeeded',
      });

      const signatureHeader = 't=123,v1=invalid_signature';

      stripeBubble = new StripeBubble({
        operation: 'handleWebhook',
        payload,
        signature: signatureHeader,
        secret: 'whsec_test_secret',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('Invalid webhook signature');
    });

    it('should handle replay attacks with timestamp validation', async () => {
      const oldTimestamp = Math.floor(Date.now() / 1000) - 1000; // 1000 seconds ago
      const payload = JSON.stringify({
        id: 'evt_test_123',
        type: 'payment_intent.succeeded',
      });

      const crypto = await import('crypto');
      const signedPayload = `${oldTimestamp}.${payload}`;
      const signature = crypto
        .createHmac('sha256', 'whsec_test_secret')
        .update(signedPayload)
        .digest('hex');
      const signatureHeader = `t=${oldTimestamp},v1=${signature}`;

      stripeBubble = new StripeBubble({
        operation: 'handleWebhook',
        payload,
        signature: signatureHeader,
        secret: 'whsec_test_secret',
        maxAge: 300000, // 5 minutes
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should handle missing credentials', async () => {
      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('API key');
    });

    it('should handle network timeouts', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Request timeout'));

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const result = await stripeBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('timeout');
    });
  });

  describe('Credential Testing', () => {
    it('should test credentials successfully', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ object: 'balance' }),
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const isValid = await stripeBubble.testCredential();

      expect(isValid).toBe(true);
    });

    it('should handle invalid credentials', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
      } as Response);

      stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 1000,
        currency: 'usd',
        credentials: { [CredentialType.STRIPE_CRED]: 'sk_invalid_key' },
      });

      const isValid = await stripeBubble.testCredential();

      expect(isValid).toBe(false);
    });
  });
});
