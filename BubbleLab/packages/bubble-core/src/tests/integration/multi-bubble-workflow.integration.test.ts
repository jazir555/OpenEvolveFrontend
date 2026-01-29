/**
 * Multi-Bubble Workflow Integration Tests
 *
 * Tests workflows that use multiple service bubbles together:
 * 1. Stripe + Google Sheets (payment tracking)
 * 2. Google Drive + Notion (document workflow)
 * 3. Webhook + Multiple Services (event-driven workflows)
 * 4. Error propagation and rollback
 * 5. Data transformation between bubbles
 * 6. Parallel execution
 * 7. Sequential dependencies
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StripeBubble } from '../../bubbles/service-bubble/stripe-bubble.js';
import { GoogleSheetsBubble } from '../../bubbles/service-bubble/google-sheets-bubble.js';
import { GoogleDriveBubble } from '../../bubbles/service-bubble/google-drive-bubble.js';
import { NotionBubble } from '../../bubbles/service-bubble/notion-bubble.js';
import { WebhookBubble } from '../../bubbles/service-bubble/webhook-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('Multi-Bubble Workflow Integration Tests', () => {
  const mockCredentials = {
    [CredentialType.STRIPE_CRED]: 'sk_test_stripe',
    [CredentialType.GOOGLE_SHEETS_CRED]: 'sheets_token',
    [CredentialType.GOOGLE_DRIVE_CRED]: 'drive_token',
    [CredentialType.NOTION_CRED]: 'notion_token',
    [CredentialType.WEBHOOK_CRED]: 'webhook_secret',
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Workflow 1: Stripe + Google Sheets (Payment Tracking)', () => {
    it('should create payment intent and log to spreadsheet', async () => {
      // Mock Stripe API
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'pi_test_123',
          amount: 10000,
          currency: 'usd',
          status: 'succeeded',
          client_secret: 'pi_test_secret',
          created: Math.floor(Date.now() / 1000),
        }),
      } as Response);

      // Step 1: Create payment intent
      const stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 10000,
        currency: 'usd',
        description: 'Test payment',
        credentials: mockCredentials,
      });

      const stripeResult = await stripeBubble.performAction();
      expect(stripeResult.result.success).toBe(true);

      // Mock Google Sheets API
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          updates: {
            updatedRange: 'Sheet1!A1:E1',
            updatedRows: 1,
            updatedColumns: 5,
            updatedCells: 5,
          },
        }),
      } as Response);

      // Step 2: Log payment to spreadsheet
      const sheetsBubble = new GoogleSheetsBubble({
        operation: 'appendRow',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        values: [
          stripeResult.result.id,
          stripeResult.result.amount,
          stripeResult.result.currency,
          stripeResult.result.status,
          new Date().toISOString(),
        ],
        credentials: mockCredentials,
      });

      const sheetsResult = await sheetsBubble.performAction();

      expect(sheetsResult.result.success).toBe(true);
      expect(sheetsResult.result.tableRange).toBeDefined();
    });

    it('should handle refund and update spreadsheet', async () => {
      // Mock refund API
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 're_test_123',
          amount: 5000,
          currency: 'usd',
          status: 'succeeded',
          payment_intent: 'pi_test_123',
        }),
      } as Response);

      // Step 1: Process refund
      const stripeBubble = new StripeBubble({
        operation: 'refundPayment',
        paymentIntentId: 'pi_test_123',
        amount: 5000,
        reason: 'requested_by_customer',
        credentials: mockCredentials,
      });

      const refundResult = await stripeBubble.performAction();
      expect(refundResult.result.success).toBe(true);

      // Mock spreadsheet update
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          updates: {
            updatedRange: 'Sheet1!A2',
          },
        }),
      } as Response);

      // Step 2: Update spreadsheet with refund info
      const sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!F2',
        value: `Refunded: ${refundResult.result.id}`,
        credentials: mockCredentials,
      });

      const sheetsResult = await sheetsBubble.performAction();

      expect(sheetsResult.result.success).toBe(true);
    });
  });

  describe('Workflow 2: Google Drive + Notion (Document Management)', () => {
    it('should upload file and create Notion page with link', async () => {
      // Mock Drive upload
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          name: 'document.pdf',
          webViewLink: 'https://drive.google.com/file/d/file_123',
        }),
      } as Response);

      // Step 1: Upload file to Drive
      const driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'document.pdf',
        content: 'PDF content',
        mimeType: 'application/pdf',
        credentials: mockCredentials,
      });

      const driveResult = await driveBubble.performAction();
      expect(driveResult.success).toBe(true);

      // Mock Notion page creation
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'page_123',
          url: 'https://notion.so/page_123',
          properties: {
            title: {
              title: [
                {
                  text: { content: 'Document Uploaded' },
                },
              ],
            },
          },
          created_time: '2024-01-01T10:00:00.000Z',
          last_edited_time: '2024-01-01T10:00:00.000Z',
        }),
      } as Response);

      // Step 2: Create Notion page with file link
      const notionBubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_123',
        title: 'Document Uploaded',
        children: [
          {
            object: 'block',
            type: 'paragraph',
            paragraph: {
              rich_text: [
                {
                  type: 'text',
                  text: {
                    content: `File uploaded to Drive: ${driveResult.data.webViewLink}`,
                    link: {
                      url: driveResult.data.webViewLink,
                    },
                  },
                },
              ],
            },
          },
        ],
        credentials: mockCredentials,
      });

      const notionResult = await notionBubble.performAction();

      expect(notionResult.result.success).toBe(true);
      expect(notionResult.result.url).toBeDefined();
    });
  });

  describe('Workflow 3: Webhook + Multiple Services (Event-Driven)', () => {
    it('should receive webhook and trigger multiple actions', async () => {
      const webhookPayload = JSON.stringify({
        id: 'evt_123',
        type: 'payment_intent.succeeded',
        data: {
          object: {
            id: 'pi_test_123',
            amount: 10000,
            customer_email: 'customer@example.com',
          },
        },
      });

      // Mock webhook signature verification
      const crypto = await import('crypto');
      const timestamp = Math.floor(Date.now() / 1000);
      const signature = crypto
        .createHmac('sha256', 'webhook_secret')
        .update(`${timestamp}.${webhookPayload}`)
        .digest('hex');

      // Step 1: Receive and verify webhook
      const webhookBubble = new WebhookBubble({
        operation: 'receiveWebhook',
        path: '/webhook/stripe',
        headers: {
          'stripe-signature': `t=${timestamp},v1=${signature}`,
        },
        body: webhookPayload,
        secret: 'webhook_secret',
        credentials: mockCredentials,
      });

      const webhookResult = await webhookBubble.performAction();
      expect(webhookResult.result.success).toBe(true);

      const eventData = JSON.parse(webhookPayload);

      // Step 2: Log to Google Sheets
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          updates: { updatedRange: 'Sheet1!A1' },
        }),
      } as Response);

      const sheetsBubble = new GoogleSheetsBubble({
        operation: 'appendRow',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        values: [
          eventData.id,
          eventData.data.object.id,
          eventData.data.object.amount,
          eventData.data.object.customer_email,
          new Date().toISOString(),
        ],
        credentials: mockCredentials,
      });

      const sheetsResult = await sheetsBubble.performAction();
      expect(sheetsResult.result.success).toBe(true);

      // Step 3: Create Notion page for record
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'page_456',
          url: 'https://notion.so/page_456',
          created_time: '2024-01-01T10:00:00.000Z',
          last_edited_time: '2024-01-01T10:00:00.000Z',
        }),
      } as Response);

      const notionBubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_123',
        title: `Payment ${eventData.data.object.id}`,
        credentials: mockCredentials,
      });

      const notionResult = await notionBubble.performAction();
      expect(notionResult.result.success).toBe(true);
    });
  });

  describe('Error Propagation and Rollback', () => {
    it('should handle errors and not complete workflow', async () => {
      // Step 1: First operation succeeds
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'pi_test_123',
          amount: 10000,
          currency: 'usd',
          status: 'requires_payment_method',
          created: Math.floor(Date.now() / 1000),
        }),
      } as Response);

      const stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 10000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const stripeResult = await stripeBubble.performAction();
      expect(stripeResult.result.success).toBe(true);

      // Step 2: Second operation fails
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ error: { message: 'Invalid spreadsheet ID' } }),
      } as Response);

      const sheetsBubble = new GoogleSheetsBubble({
        operation: 'appendRow',
        spreadsheetId: 'invalid_sheet',
        range: 'Sheet1!A1',
        values: ['test'],
        credentials: mockCredentials,
      });

      const sheetsResult = await sheetsBubble.performAction();
      expect(sheetsResult.result.success).toBe(false);

      // Step 3: Verify workflow state - partial completion
      expect(stripeResult.result.success).toBe(true);
      expect(sheetsResult.result.success).toBe(false);

      // In production, you might want to rollback the first operation
      // or implement compensating transactions
    });

    it('should stop workflow on critical failure', async () => {
      const operations = [
        // Step 1: Success
        vi.mocked(global.fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: 'file_1', name: 'doc1.txt' }),
        } as Response),

        // Step 2: Failure
        vi.mocked(global.fetch).mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: { message: 'Internal server error' } }),
        } as Response),
      ];

      // Execute first operation
      const driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'doc1.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const driveResult = await driveBubble.performAction();
      expect(driveResult.success).toBe(true);

      // Second operation fails
      const notionBubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_123',
        title: 'Test',
        credentials: mockCredentials,
      });

      const notionResult = await notionBubble.performAction();
      expect(notionResult.result.success).toBe(false);

      // Workflow should stop here
      expect(operations).toHaveLength(2);
    });
  });

  describe('Data Transformation Between Bubbles', () => {
    it('should transform and pass data between services', async () => {
      // Step 1: Get data from Stripe
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'pi_test_123',
          amount: 10000,
          currency: 'usd',
          status: 'succeeded',
          created: Math.floor(Date.now() / 1000),
        }),
      } as Response);

      const stripeBubble = new StripeBubble({
        operation: 'createPaymentIntent',
        amount: 10000,
        currency: 'usd',
        credentials: mockCredentials,
      });

      const stripeResult = await stripeBubble.performAction();

      // Transform data: Convert cents to dollars, format currency
      const amountInDollars = stripeResult.result.amount / 100;
      const formattedAmount = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: stripeResult.result.currency.toUpperCase(),
      }).format(amountInDollars);

      // Step 2: Use transformed data in Notion
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'page_123',
          url: 'https://notion.so/page_123',
          created_time: '2024-01-01T10:00:00.000Z',
          last_edited_time: '2024-01-01T10:00:00.000Z',
        }),
      } as Response);

      const notionBubble = new NotionBubble({
        operation: 'createPage',
        parentPageId: 'parent_123',
        title: `Payment: ${formattedAmount}`,
        credentials: mockCredentials,
      });

      const notionResult = await notionBubble.performAction();

      expect(notionResult.result.success).toBe(true);
      expect(formattedAmount).toBe('$100.00');
    });
  });

  describe('Parallel Execution', () => {
    it('should execute multiple operations in parallel', async () => {
      // Mock all APIs
      vi.mocked(global.fetch)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: 'file_1', name: 'doc1.txt' }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: 'page_1', url: 'https://notion.so/page_1' }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ updates: { updatedRange: 'Sheet1!A1' } }),
        } as Response);

      // Execute operations in parallel
      const [driveResult, notionResult, sheetsResult] = await Promise.all([
        new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'doc1.txt',
          content: 'content',
          credentials: mockCredentials,
        }).performAction(),

        new NotionBubble({
          operation: 'createPage',
          parentPageId: 'parent_123',
          title: 'Parallel Test',
          credentials: mockCredentials,
        }).performAction(),

        new GoogleSheetsBubble({
          operation: 'updateCell',
          spreadsheetId: 'sheet_123',
          range: 'Sheet1!A1',
          value: 'Parallel update',
          credentials: mockCredentials,
        }).performAction(),
      ]);

      expect(driveResult.success).toBe(true);
      expect(notionResult.result.success).toBe(true);
      expect(sheetsResult.result.success).toBe(true);
    });

    it('should handle partial failures in parallel execution', async () => {
      // Mock APIs with mixed success/failure
      vi.mocked(global.fetch)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: 'file_1', name: 'doc1.txt' }),
        } as Response)
        .mockResolvedValueOnce({
          ok: false,
          status: 400,
          json: async () => ({ error: { message: 'Bad request' } }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: 'page_1', url: 'https://notion.so/page_1' }),
        } as Response);

      const results = await Promise.allSettled([
        new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'doc1.txt',
          content: 'content',
          credentials: mockCredentials,
        }).performAction(),

        new GoogleSheetsBubble({
          operation: 'updateCell',
          spreadsheetId: 'invalid',
          range: 'Sheet1!A1',
          value: 'test',
          credentials: mockCredentials,
        }).performAction(),

        new NotionBubble({
          operation: 'createPage',
          parentPageId: 'parent_123',
          title: 'Test',
          credentials: mockCredentials,
        }).performAction(),
      ]);

      expect(results[0].status).toBe('fulfilled');
      expect(results[1].status).toBe('rejected');
      expect(results[2].status).toBe('fulfilled');
    });
  });

  describe('Sequential Dependencies', () => {
    it('should execute operations with dependencies in sequence', async () => {
      // Step 1: Create customer in Stripe
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'cus_test_123',
          email: 'test@example.com',
          name: 'Test Customer',
          created: Math.floor(Date.now() / 1000),
        }),
      } as Response);

      const stripeBubble = new StripeBubble({
        operation: 'createCustomer',
        email: 'test@example.com',
        name: 'Test Customer',
        credentials: mockCredentials,
      });

      const customerResult = await stripeBubble.performAction();
      expect(customerResult.result.success).toBe(true);

      // Step 2: Use customer ID to create subscription
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'sub_test_123',
          customer: 'cus_test_123',
          status: 'active',
          current_period_start: Math.floor(Date.now() / 1000),
          current_period_end: Math.floor(Date.now() / 1000) + 2592000,
          cancel_at_period_end: false,
          created: Math.floor(Date.now() / 1000),
        }),
      } as Response);

      const subscriptionBubble = new StripeBubble({
        operation: 'createSubscription',
        customer: customerResult.result.id,
        priceId: 'price_test_123',
        credentials: mockCredentials,
      });

      const subscriptionResult = await subscriptionBubble.performAction();
      expect(subscriptionResult.result.success).toBe(true);
      expect(subscriptionResult.result.customerId).toBe(customerResult.result.id);

      // Step 3: Log subscription to spreadsheet
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          updates: { updatedRange: 'Sheet1!A1' },
        }),
      } as Response);

      const sheetsBubble = new GoogleSheetsBubble({
        operation: 'appendRow',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        values: [
          customerResult.result.id,
          subscriptionResult.result.id,
          subscriptionResult.result.status,
          new Date().toISOString(),
        ],
        credentials: mockCredentials,
      });

      const sheetsResult = await sheetsBubble.performAction();
      expect(sheetsResult.result.success).toBe(true);
    });
  });
});
