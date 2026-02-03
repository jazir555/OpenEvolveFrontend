import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import {
  ResilienceWrapper,
  DEFAULT_RESILIENCE_CONFIG,
} from '../../__mocks__/resilience.js';
import {
  validateEmail,
  validateNonEmptyString,
  validateNumberRange,
  ValidationError as CommonValidationError
} from '../common/validators.js';
import {
  AuthenticationError,
  ExternalServiceError,
  ValidationError,
  NotFoundError,
  createErrorResponse
} from '../common/error-handlers.js';

/**
 * Stripe Bubble - Complete Service Bubble Implementation
 *
 * Full production implementation with 15 operations:
 * 1. createPaymentIntent - Create a payment intent for one-time payments
 * 2. confirmPayment - Confirm a payment intent
 * 3. refundPayment - Create a refund for a payment
 * 4. createCustomer - Create a new customer
 * 5. getCustomer - Retrieve customer details
 * 6. updateCustomer - Update customer information
 * 7. createSubscription - Create a new subscription
 * 8. cancelSubscription - Cancel a subscription
 * 9. updateSubscription - Update subscription details
 * 10. createInvoice - Create an invoice
 * 11. getInvoice - Retrieve invoice details
 * 12. listInvoices - List invoices with pagination
 * 13. createProduct - Create a product
 * 14. createPrice - Create a price for a product
 * 15. handleWebhook - Verify and process Stripe webhooks
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const CreatePaymentIntentParamsSchema = z.object({
  operation: z.literal('createPaymentIntent'),
  amount: z.number().positive().describe('Amount in cents (e.g., $10.00 = 1000)'),
  currency: z.string().length(3).default('usd').describe('3-letter currency code'),
  customer: z.string().optional().describe('Customer ID'),
  paymentMethod: z.string().optional().describe('Payment method ID'),
  description: z.string().optional(),
  metadata: z.record(z.string()).optional(),
  confirm: z.boolean().optional().default(false),
  captureMethod: z.enum(['automatic', 'manual']).optional().default('automatic'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ConfirmPaymentParamsSchema = z.object({
  operation: z.literal('confirmPayment'),
  paymentIntentId: z.string().min(1, 'Payment Intent ID is required'),
  paymentMethod: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const RefundPaymentParamsSchema = z.object({
  operation: z.literal('refundPayment'),
  paymentIntentId: z.string().min(1, 'Payment Intent ID is required'),
  amount: z.number().positive().optional().describe('Amount to refund in cents'),
  reason: z.enum(['duplicate', 'fraudulent', 'requested_by_customer', 'other']).optional(),
  metadata: z.record(z.string()).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateCustomerParamsSchema = z.object({
  operation: z.literal('createCustomer'),
  email: z.string().email().optional(),
  name: z.string().optional(),
  phone: z.string().optional(),
  description: z.string().optional(),
  metadata: z.record(z.string()).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetCustomerParamsSchema = z.object({
  operation: z.literal('getCustomer'),
  customerId: z.string().min(1, 'Customer ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateCustomerParamsSchema = z.object({
  operation: z.literal('updateCustomer'),
  customerId: z.string().min(1, 'Customer ID is required'),
  email: z.string().email().optional(),
  name: z.string().optional(),
  phone: z.string().optional(),
  description: z.string().optional(),
  metadata: z.record(z.string()).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateSubscriptionParamsSchema = z.object({
  operation: z.literal('createSubscription'),
  customer: z.string().min(1, 'Customer ID is required'),
  priceId: z.string().min(1, 'Price ID is required'),
  quantity: z.number().int().positive().optional().default(1),
  trialPeriodDays: z.number().int().nonnegative().optional(),
  metadata: z.record(z.string()).optional(),
  paymentBehavior: z.enum(['default_incomplete', 'allow_incomplete', 'error_if_incomplete']).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CancelSubscriptionParamsSchema = z.object({
  operation: z.literal('cancelSubscription'),
  subscriptionId: z.string().min(1, 'Subscription ID is required'),
  cancelAtPeriodEnd: z.boolean().optional().default(true),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateSubscriptionParamsSchema = z.object({
  operation: z.literal('updateSubscription'),
  subscriptionId: z.string().min(1, 'Subscription ID is required'),
  priceId: z.string().optional().describe('New price ID for the subscription'),
  quantity: z.number().int().positive().optional(),
  metadata: z.record(z.string()).optional(),
  prorationBehavior: z.enum(['create_prorations', 'always_invoice', 'none']).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateInvoiceParamsSchema = z.object({
  operation: z.literal('createInvoice'),
  customer: z.string().min(1, 'Customer ID is required'),
  description: z.string().optional(),
  metadata: z.record(z.string()).optional(),
  autoAdvance: z.boolean().optional().default(true),
  collectionMethod: z.enum(['charge_automatically', 'send_invoice']).optional().default('charge_automatically'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetInvoiceParamsSchema = z.object({
  operation: z.literal('getInvoice'),
  invoiceId: z.string().min(1, 'Invoice ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ListInvoicesParamsSchema = z.object({
  operation: z.literal('listInvoices'),
  customer: z.string().optional().describe('Filter by customer ID'),
  limit: z.number().int().positive().optional().default(10),
  startingAfter: z.string().optional().describe('Pagination cursor'),
  status: z.enum(['draft', 'open', 'paid', 'uncollectible', 'void']).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateProductParamsSchema = z.object({
  operation: z.literal('createProduct'),
  name: z.string().min(1, 'Product name is required'),
  description: z.string().optional(),
  metadata: z.record(z.string()).optional(),
  images: z.array(z.string().url()).optional(),
  statementDescriptor: z.string().optional(),
  unitLabel: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreatePriceParamsSchema = z.object({
  operation: z.literal('createPrice'),
  product: z.string().min(1, 'Product ID is required'),
  unitAmount: z.number().positive().describe('Amount in cents'),
  currency: z.string().length(3).default('usd'),
  recurring: z.object({
    interval: z.enum(['day', 'week', 'month', 'year']),
    intervalCount: z.number().int().positive().optional().default(1),
    usageType: z.enum(['licensed', 'metered']).optional().default('licensed'),
  }).optional(),
  nickname: z.string().optional(),
  metadata: z.record(z.string()).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const HandleWebhookParamsSchema = z.object({
  operation: z.literal('handleWebhook'),
  payload: z.string().describe('Raw webhook payload string'),
  signature: z.string().describe('Stripe signature from headers'),
  secret: z.string().min(1, 'Webhook signing secret'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const StripeBubbleParamsSchema = z.discriminatedUnion('operation', [
  CreatePaymentIntentParamsSchema,
  ConfirmPaymentParamsSchema,
  RefundPaymentParamsSchema,
  CreateCustomerParamsSchema,
  GetCustomerParamsSchema,
  UpdateCustomerParamsSchema,
  CreateSubscriptionParamsSchema,
  CancelSubscriptionParamsSchema,
  UpdateSubscriptionParamsSchema,
  CreateInvoiceParamsSchema,
  GetInvoiceParamsSchema,
  ListInvoicesParamsSchema,
  CreateProductParamsSchema,
  CreatePriceParamsSchema,
  HandleWebhookParamsSchema,
]);

export type StripeBubbleParams = z.input<typeof StripeBubbleParamsSchema>;

// ============================================================================
// RESULT SCHEMAS
// ============================================================================

const PaymentIntentResultSchema = z.object({
  id: z.string(),
  amount: z.number(),
  currency: z.string(),
  status: z.string(),
  clientSecret: z.string().optional(),
  description: z.string().optional(),
  createdAt: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const RefundResultSchema = z.object({
  id: z.string(),
  amount: z.number(),
  currency: z.string(),
  status: z.string(),
  paymentIntentId: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const CustomerResultSchema = z.object({
  id: z.string(),
  email: z.string().optional(),
  name: z.string().optional(),
  phone: z.string().optional(),
  description: z.string().optional(),
  createdAt: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const SubscriptionResultSchema = z.object({
  id: z.string(),
  customerId: z.string(),
  status: z.string(),
  currentPeriodStart: z.string(),
  currentPeriodEnd: z.string(),
  cancelAtPeriodEnd: z.boolean(),
  createdAt: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const InvoiceResultSchema = z.object({
  id: z.string(),
  number: z.string().optional(),
  status: z.string(),
  amountDue: z.number(),
  currency: z.string(),
  customer: z.string(),
  createdAt: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const InvoiceListSchema = z.object({
  invoices: z.array(z.object({
    id: z.string(),
    number: z.string().optional(),
    status: z.string(),
    amountDue: z.number(),
    currency: z.string(),
    customer: z.string(),
    createdAt: z.string(),
  })),
  hasMore: z.boolean(),
  count: z.number(),
  success: z.boolean(),
  error: z.string(),
});

const ProductResultSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  active: z.boolean().optional(),
  createdAt: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const PriceResultSchema = z.object({
  id: z.string(),
  productId: z.string(),
  unitAmount: z.number(),
  currency: z.string(),
  recurring: z.any().optional(),
  active: z.boolean(),
  createdAt: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const WebhookEventSchema = z.object({
  id: z.string(),
  type: z.string(),
  data: z.any(),
  processed: z.boolean(),
  success: z.boolean(),
  error: z.string(),
});

const StripeBubbleResultSchema = z.discriminatedUnion('operation', [
  z.object({
    operation: z.literal('createPaymentIntent'),
    result: PaymentIntentResultSchema,
  }),
  z.object({
    operation: z.literal('confirmPayment'),
    result: PaymentIntentResultSchema,
  }),
  z.object({
    operation: z.literal('refundPayment'),
    result: RefundResultSchema,
  }),
  z.object({
    operation: z.literal('createCustomer'),
    result: CustomerResultSchema,
  }),
  z.object({
    operation: z.literal('getCustomer'),
    result: CustomerResultSchema,
  }),
  z.object({
    operation: z.literal('updateCustomer'),
    result: CustomerResultSchema,
  }),
  z.object({
    operation: z.literal('createSubscription'),
    result: SubscriptionResultSchema,
  }),
  z.object({
    operation: z.literal('cancelSubscription'),
    result: SubscriptionResultSchema,
  }),
  z.object({
    operation: z.literal('updateSubscription'),
    result: SubscriptionResultSchema,
  }),
  z.object({
    operation: z.literal('createInvoice'),
    result: InvoiceResultSchema,
  }),
  z.object({
    operation: z.literal('getInvoice'),
    result: InvoiceResultSchema,
  }),
  z.object({
    operation: z.literal('listInvoices'),
    result: InvoiceListSchema,
  }),
  z.object({
    operation: z.literal('createProduct'),
    result: ProductResultSchema,
  }),
  z.object({
    operation: z.literal('createPrice'),
    result: PriceResultSchema,
  }),
  z.object({
    operation: z.literal('handleWebhook'),
    result: WebhookEventSchema,
  }),
]);

type StripeBubbleResult = z.output<typeof StripeBubbleResultSchema>;

// ============================================================================
// STRIPE API CLIENT
// ============================================================================

/**
 * Stripe API Client
 * Handles HTTP communication with the Stripe API
 */
class StripeClient {
  private baseUrl: string = 'https://api.stripe.com/v1';
  private headers: Record<string, string>;

  /**
   * Create a new Stripe API client
   * @param apiKey - Stripe API key for authentication
   */
  constructor(apiKey: string) {
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    };
  }

  /**
   * Encode parameters for URL-encoded form data
   * @param params - Parameters to encode
   * @returns URL-encoded parameter string
   */
  private encodeParams(params: Record<string, any>): string {
    return Object.entries(params)
      .filter(([_, value]) => value !== undefined && value !== null)
      .map(([key, value]) => {
        if (typeof value === 'object') {
          return `${key}=${encodeURIComponent(JSON.stringify(value))}`;
        }
        return `${key}=${encodeURIComponent(String(value))}`;
      })
      .join('&');
  }

  /**
   * Make a GET request to the Stripe API
   * @param endpoint - API endpoint (e.g., 'customers', 'payment_intents')
   * @param params - Query parameters
   * @returns Promise that resolves with the API response
   * @throws ExternalServiceError if the API request fails
   */
  async get(endpoint: string, params?: Record<string, any>): Promise<any> {
    const url = new URL(`${this.baseUrl}/${endpoint}`);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Authorization': this.headers.Authorization,
      },
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new ExternalServiceError('stripe', `GET ${endpoint} failed`, String(response.status), { error });
    }

    return response.json();
  }

  /**
   * Make a POST request to the Stripe API with form-encoded data
   * @param endpoint - API endpoint
   * @param params - Request body parameters
   * @returns Promise that resolves with the API response
   * @throws ExternalServiceError if the API request fails
   */
  async post(endpoint: string, params?: Record<string, any>): Promise<any> {
    const url = `${this.baseUrl}/${endpoint}`;
    const body = params ? this.encodeParams(params) : '';

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body,
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new ExternalServiceError('stripe', `POST ${endpoint} failed`, String(response.status), { error });
    }

    return response.json();
  }

  /**
   * Make a POST request to the Stripe API with JSON data
   * @param endpoint - API endpoint
   * @param params - Request body parameters (JSON)
   * @returns Promise that resolves with the API response
   * @throws ExternalServiceError if the API request fails
   */
  async postJson(endpoint: string, params: Record<string, any>): Promise<any> {
    const url = `${this.baseUrl}/${endpoint}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': this.headers.Authorization,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new ExternalServiceError('stripe', `POST ${endpoint} (JSON) failed`, String(response.status), { error });
    }

    return response.json();
  }

  /**
   * Make a DELETE request to the Stripe API
   * @param endpoint - API endpoint
   * @returns Promise that resolves with the API response
   * @throws ExternalServiceError if the API request fails
   */
  async delete(endpoint: string): Promise<any> {
    const url = `${this.baseUrl}/${endpoint}`;
    const response = await fetch(url, {
      method: 'DELETE',
      headers: {
        'Authorization': this.headers.Authorization,
      },
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new ExternalServiceError('stripe', `DELETE ${endpoint} failed`, String(response.status), { error });
    }

    return response.json();
  }
}

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

/**
 * Stripe Bubble - Complete Service Bubble Implementation
 *
 * Provides comprehensive integration with the Stripe API for payment processing,
 * customer management, subscriptions, invoicing, and webhook handling.
 *
 * @template T - Stripe bubble parameters type
 */
export class StripeBubble<
  T extends StripeBubbleParams = StripeBubbleParams
> extends ServiceBubble<T, any> {
  static readonly type = 'service' as const;
  static readonly service = 'stripe';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'stripe';
  static readonly schema = StripeBubbleParamsSchema;
  static readonly resultSchema = StripeBubbleResultSchema;
  static readonly shortDescription = 'Complete Stripe integration for payments and billing';
  static readonly longDescription = `
    Comprehensive Stripe service bubble for all payment operations.

    Operations:
    1. createPaymentIntent - Create payment intents for one-time payments
    2. confirmPayment - Confirm and process payment intents
    3. refundPayment - Create refunds for payments
    4. createCustomer - Create new customer records
    5. getCustomer - Retrieve customer information
    6. updateCustomer - Update customer details
    7. createSubscription - Create recurring subscriptions
    8. cancelSubscription - Cancel subscriptions
    9. updateSubscription - Modify subscription details
    10. createInvoice - Create and send invoices
    11. getInvoice - Retrieve invoice details
    12. listInvoices - List customer invoices
    13. createProduct - Create products
    14. createPrice - Create product prices
    15. handleWebhook - Verify and process webhooks

    Features:
    - Full payment lifecycle management
    - Subscription and recurring billing
    - Invoice generation and management
    - Product and price management
    - Webhook signature verification
    - Customer management
    - Refund processing
    - Resilience patterns with automatic retries
  `;
  static readonly alias = 'stripe';

  private client: StripeClient | null = null;
  private resilience: ResilienceWrapper;

  /**
   * Create a new Stripe Bubble instance
   * @param params - Operation parameters
   * @param context - Bubble execution context
   */
  constructor(
    params: T,
    context?: BubbleContext
  ) {
    super(params, context);

    this.resilience = new ResilienceWrapper(
      DEFAULT_RESILIENCE_CONFIG
    );
  }

  /**
   * Test the validity of the Stripe API credentials
   * @returns Promise that resolves to true if credentials are valid, false otherwise
   */
  public async testCredential(): Promise<boolean> {
    const apiKey = this.chooseCredential();
    if (!apiKey) {
      return false;
    }

    try {
      const client = new StripeClient(apiKey);
      await client.get('balance');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Extract the Stripe API key from credentials
   * @returns Stripe API key or undefined if not found
   * @throws AuthenticationError if credentials are invalid or missing
   */
  protected chooseCredential(): string | undefined {
    const credentials = (this.params as any).credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new AuthenticationError('Stripe API credentials are required');
    }
    return credentials[CredentialType.STRIPE_CRED];
  }

  /**
   * Execute the Stripe operation specified in params
   * @param context - Bubble execution context (unused)
   * @returns Promise that resolves with the operation result
   * @throws AuthenticationError if API key is missing
   */
  protected async performAction(
    context?: BubbleContext
  ): Promise<Extract<StripeBubbleResult, { operation: T['operation'] }>> {
    void context;

    const apiKey = this.chooseCredential();
    if (!apiKey) {
      return this.errorResult('Stripe API key is required');
    }

    this.client = new StripeClient(apiKey);

    const { operation } = this.params;

    try {
      const result = await this.resilience.execute(
        `stripe-${operation}-${Date.now()}`,
        async () => {
          switch (operation) {
            case 'createPaymentIntent':
              return await this.createPaymentIntent(this.params as any);
            case 'confirmPayment':
              return await this.confirmPayment(this.params as any);
            case 'refundPayment':
              return await this.refundPayment(this.params as any);
            case 'createCustomer':
              return await this.createCustomer(this.params as any);
            case 'getCustomer':
              return await this.getCustomer(this.params as any);
            case 'updateCustomer':
              return await this.updateCustomer(this.params as any);
            case 'createSubscription':
              return await this.createSubscription(this.params as any);
            case 'cancelSubscription':
              return await this.cancelSubscription(this.params as any);
            case 'updateSubscription':
              return await this.updateSubscription(this.params as any);
            case 'createInvoice':
              return await this.createInvoice(this.params as any);
            case 'getInvoice':
              return await this.getInvoice(this.params as any);
            case 'listInvoices':
              return await this.listInvoices(this.params as any);
            case 'createProduct':
              return await this.createProduct(this.params as any);
            case 'createPrice':
              return await this.createPrice(this.params as any);
            case 'handleWebhook':
              return await this.handleWebhook(this.params as any);
            default:
              throw new Error(`Unsupported operation: ${operation}`);
          }
        }
      );

      return {
        operation,
        result,
      } as any;
    } catch (error) {
      return {
        operation,
        result: {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        },
      } as any;
    }
  }

  // ========================================================================
  // OPERATION 1: CREATE PAYMENT INTENT
  // ========================================================================

  /**
   * Create a payment intent for one-time payment
   * @param params - Payment intent parameters
   * @returns Promise that resolves with the payment intent result
   */
  private async createPaymentIntent(
    params: Extract<StripeBubbleParams, { operation: 'createPaymentIntent' }>
  ): Promise<typeof PaymentIntentResultSchema._output> {
    const { amount, currency, customer, paymentMethod, description, metadata, confirm, captureMethod } = params;

    try {
      const response = await this.client!.post('payment_intents', {
        amount,
        currency,
        customer,
        payment_method: paymentMethod,
        description,
        metadata,
        confirm,
        capture_method: captureMethod,
      });

      return {
        id: response.id,
        amount: response.amount,
        currency: response.currency,
        status: response.status,
        clientSecret: response.client_secret,
        description: response.description,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: '',
        amount,
        currency: currency!,
        status: '',
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create payment intent',
      };
    }
  }

  // ========================================================================
  // OPERATION 2: CONFIRM PAYMENT
  // ========================================================================

  /**
   * Confirm a payment intent
   * @param params - Confirm payment parameters
   * @returns Promise that resolves with the payment intent result
   */
  private async confirmPayment(
    params: Extract<StripeBubbleParams, { operation: 'confirmPayment' }>
  ): Promise<typeof PaymentIntentResultSchema._output> {
    const { paymentIntentId, paymentMethod } = params;

    try {
      const response = await this.client!.post(`payment_intents/${paymentIntentId}/confirm`, {
        payment_method: paymentMethod,
      });

      return {
        id: response.id,
        amount: response.amount,
        currency: response.currency,
        status: response.status,
        clientSecret: response.client_secret,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: paymentIntentId,
        amount: 0,
        currency: '',
        status: '',
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to confirm payment',
      };
    }
  }

  // ========================================================================
  // OPERATION 3: REFUND PAYMENT
  // ========================================================================

  /**
   * Create a refund for a payment
   * @param params - Refund parameters
   * @returns Promise that resolves with the refund result
   */
  private async refundPayment(
    params: Extract<StripeBubbleParams, { operation: 'refundPayment' }>
  ): Promise<typeof RefundResultSchema._output> {
    const { paymentIntentId, amount, reason, metadata } = params;

    try {
      const response = await this.client!.post('refunds', {
        payment_intent: paymentIntentId,
        amount,
        reason,
        metadata,
      });

      return {
        id: response.id,
        amount: response.amount,
        currency: response.currency,
        status: response.status,
        paymentIntentId,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: '',
        amount: amount || 0,
        currency: '',
        status: '',
        paymentIntentId,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to refund payment',
      };
    }
  }

  // ========================================================================
  // OPERATION 4: CREATE CUSTOMER
  // ========================================================================

  /**
   * Create a new customer in Stripe
   * @param params - Customer creation parameters
   * @returns Promise that resolves with the customer result
   */
  private async createCustomer(
    params: Extract<StripeBubbleParams, { operation: 'createCustomer' }>
  ): Promise<typeof CustomerResultSchema._output> {
    const { email, name, phone, description, metadata } = params;

    try {
      const response = await this.client!.post('customers', {
        email,
        name,
        phone,
        description,
        metadata,
      });

      return {
        id: response.id,
        email: response.email,
        name: response.name,
        phone: response.phone,
        description: response.description,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: '',
        email: email || '',
        name: name || '',
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create customer',
      };
    }
  }

  // ========================================================================
  // OPERATION 5: GET CUSTOMER
  // ========================================================================

  private async getCustomer(
    params: Extract<StripeBubbleParams, { operation: 'getCustomer' }>
  ): Promise<typeof CustomerResultSchema._output> {
    const { customerId } = params;

    try {
      const response = await this.client!.get(`customers/${customerId}`);

      return {
        id: response.id,
        email: response.email,
        name: response.name,
        phone: response.phone,
        description: response.description,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: customerId,
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get customer',
      };
    }
  }

  // ========================================================================
  // OPERATION 6: UPDATE CUSTOMER
  // ========================================================================

  private async updateCustomer(
    params: Extract<StripeBubbleParams, { operation: 'updateCustomer' }>
  ): Promise<typeof CustomerResultSchema._output> {
    const { customerId, email, name, phone, description, metadata } = params;

    try {
      const response = await this.client!.post(`customers/${customerId}`, {
        email,
        name,
        phone,
        description,
        metadata,
      });

      return {
        id: response.id,
        email: response.email,
        name: response.name,
        phone: response.phone,
        description: response.description,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: customerId,
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update customer',
      };
    }
  }

  // ========================================================================
  // OPERATION 7: CREATE SUBSCRIPTION
  // ========================================================================

  private async createSubscription(
    params: Extract<StripeBubbleParams, { operation: 'createSubscription' }>
  ): Promise<typeof SubscriptionResultSchema._output> {
    const { customer, priceId, quantity, trialPeriodDays, metadata, paymentBehavior } = params;

    try {
      const response = await this.client!.post('subscriptions', {
        customer,
        items: [{
          price: priceId,
          quantity,
        }],
        trial_period_days: trialPeriodDays,
        metadata,
        payment_behavior: paymentBehavior,
      });

      return {
        id: response.id,
        customerId: response.customer,
        status: response.status,
        currentPeriodStart: new Date(response.current_period_start * 1000).toISOString(),
        currentPeriodEnd: new Date(response.current_period_end * 1000).toISOString(),
        cancelAtPeriodEnd: response.cancel_at_period_end,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: '',
        customerId: customer,
        status: '',
        currentPeriodStart: '',
        currentPeriodEnd: '',
        cancelAtPeriodEnd: false,
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create subscription',
      };
    }
  }

  // ========================================================================
  // OPERATION 8: CANCEL SUBSCRIPTION
  // ========================================================================

  private async cancelSubscription(
    params: Extract<StripeBubbleParams, { operation: 'cancelSubscription' }>
  ): Promise<typeof SubscriptionResultSchema._output> {
    const { subscriptionId, cancelAtPeriodEnd } = params;

    try {
      const response = await this.client!.delete(`subscriptions/${subscriptionId}`);

      return {
        id: response.id,
        customerId: response.customer,
        status: response.status,
        currentPeriodStart: new Date(response.current_period_start * 1000).toISOString(),
        currentPeriodEnd: new Date(response.current_period_end * 1000).toISOString(),
        cancelAtPeriodEnd: response.cancel_at_period_end,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: subscriptionId,
        customerId: '',
        status: '',
        currentPeriodStart: '',
        currentPeriodEnd: '',
        cancelAtPeriodEnd: cancelAtPeriodEnd!,
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to cancel subscription',
      };
    }
  }

  // ========================================================================
  // OPERATION 9: UPDATE SUBSCRIPTION
  // ========================================================================

  private async updateSubscription(
    params: Extract<StripeBubbleParams, { operation: 'updateSubscription' }>
  ): Promise<typeof SubscriptionResultSchema._output> {
    const { subscriptionId, priceId, quantity, metadata, prorationBehavior } = params;

    try {
      const updateData: any = {
        metadata,
        proration_behavior: prorationBehavior,
      };

      if (priceId) {
        updateData.items = [{
          price: priceId,
          quantity,
        }];
      }

      const response = await this.client!.post(`subscriptions/${subscriptionId}`, updateData);

      return {
        id: response.id,
        customerId: response.customer,
        status: response.status,
        currentPeriodStart: new Date(response.current_period_start * 1000).toISOString(),
        currentPeriodEnd: new Date(response.current_period_end * 1000).toISOString(),
        cancelAtPeriodEnd: response.cancel_at_period_end,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: subscriptionId,
        customerId: '',
        status: '',
        currentPeriodStart: '',
        currentPeriodEnd: '',
        cancelAtPeriodEnd: false,
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update subscription',
      };
    }
  }

  // ========================================================================
  // OPERATION 10: CREATE INVOICE
  // ========================================================================

  private async createInvoice(
    params: Extract<StripeBubbleParams, { operation: 'createInvoice' }>
  ): Promise<typeof InvoiceResultSchema._output> {
    const { customer, description, metadata, autoAdvance, collectionMethod } = params;

    try {
      const response = await this.client!.post('invoices', {
        customer,
        description,
        metadata,
        auto_advance: autoAdvance,
        collection_method: collectionMethod,
      });

      return {
        id: response.id,
        number: response.number,
        status: response.status,
        amountDue: response.amount_due,
        currency: response.currency,
        customer: response.customer,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: '',
        status: '',
        amountDue: 0,
        currency: '',
        customer,
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create invoice',
      };
    }
  }

  // ========================================================================
  // OPERATION 11: GET INVOICE
  // ========================================================================

  private async getInvoice(
    params: Extract<StripeBubbleParams, { operation: 'getInvoice' }>
  ): Promise<typeof InvoiceResultSchema._output> {
    const { invoiceId } = params;

    try {
      const response = await this.client!.get(`invoices/${invoiceId}`);

      return {
        id: response.id,
        number: response.number,
        status: response.status,
        amountDue: response.amount_due,
        currency: response.currency,
        customer: response.customer,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: invoiceId,
        status: '',
        amountDue: 0,
        currency: '',
        customer: '',
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get invoice',
      };
    }
  }

  // ========================================================================
  // OPERATION 12: LIST INVOICES
  // ========================================================================

  private async listInvoices(
    params: Extract<StripeBubbleParams, { operation: 'listInvoices' }>
  ): Promise<typeof InvoiceListSchema._output> {
    const { customer, limit, startingAfter, status } = params;

    try {
      const queryParams: Record<string, any> = {
        limit,
      };

      if (customer) {
        queryParams.customer = customer;
      }

      if (startingAfter) {
        queryParams.starting_after = startingAfter;
      }

      if (status) {
        queryParams.status = status;
      }

      const response = await this.client!.get('invoices', queryParams);

      return {
        invoices: response.data.map((invoice: any) => ({
          id: invoice.id,
          number: invoice.number,
          status: invoice.status,
          amountDue: invoice.amount_due,
          currency: invoice.currency,
          customer: invoice.customer,
          createdAt: new Date(invoice.created * 1000).toISOString(),
        })),
        hasMore: response.has_more,
        count: response.data.length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        invoices: [],
        hasMore: false,
        count: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to list invoices',
      };
    }
  }

  // ========================================================================
  // OPERATION 13: CREATE PRODUCT
  // ========================================================================

  private async createProduct(
    params: Extract<StripeBubbleParams, { operation: 'createProduct' }>
  ): Promise<typeof ProductResultSchema._output> {
    const { name, description, metadata, images, statementDescriptor, unitLabel } = params;

    try {
      const response = await this.client!.post('products', {
        name,
        description,
        metadata,
        images,
        statement_descriptor: statementDescriptor,
        unit_label: unitLabel,
      });

      return {
        id: response.id,
        name: response.name,
        description: response.description,
        active: response.active,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: '',
        name,
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create product',
      };
    }
  }

  // ========================================================================
  // OPERATION 14: CREATE PRICE
  // ========================================================================

  private async createPrice(
    params: Extract<StripeBubbleParams, { operation: 'createPrice' }>
  ): Promise<typeof PriceResultSchema._output> {
    const { product, unitAmount, currency, recurring, nickname, metadata } = params;

    try {
      const response = await this.client!.post('prices', {
        product,
        unit_amount: unitAmount,
        currency,
        recurring,
        nickname,
        metadata,
      });

      return {
        id: response.id,
        productId: response.product,
        unitAmount: response.unit_amount,
        currency: response.currency,
        recurring: response.recurring,
        active: response.active,
        createdAt: new Date(response.created * 1000).toISOString(),
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: '',
        productId: product,
        unitAmount,
        currency: currency!,
        active: false,
        createdAt: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create price',
      };
    }
  }

  // ========================================================================
  // OPERATION 15: HANDLE WEBHOOK
  // ========================================================================

  private async handleWebhook(
    params: Extract<StripeBubbleParams, { operation: 'handleWebhook' }>
  ): Promise<typeof WebhookEventSchema._output> {
    const { payload, signature, secret } = params;

    try {
      // Verify webhook signature
      const crypto = await import('crypto');
      const elements = signature.split(',');
      const timestamp = elements[0].split('=')[1];
      const signatures = elements.slice(1).map(e => e.split('=')[1]);

      const signedPayload = `${timestamp}.${payload}`;
      const expectedSignature = crypto
        .createHmac('sha256', secret)
        .update(signedPayload)
        .digest('hex');

      const signatureValid = signatures.some(sig =>
        crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(expectedSignature))
      );

      if (!signatureValid) {
        return {
          id: '',
          type: '',
          data: null,
          processed: false,
          success: false,
          error: 'Invalid webhook signature',
        };
      }

      // Parse webhook payload
      const event = JSON.parse(payload);

      return {
        id: event.id,
        type: event.type,
        data: event.data,
        processed: true,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        id: '',
        type: '',
        data: null,
        processed: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to handle webhook',
      };
    }
  }

  // ========================================================================
  // HELPER METHODS
  // ========================================================================

  /**
   * Create an error result object
   * @param error - Error message
   * @returns Error result object
   */
  private errorResult(error: string): any {
    return {
      operation: this.params.operation,
      result: {
        success: false,
        error,
      },
    };
  }
}
