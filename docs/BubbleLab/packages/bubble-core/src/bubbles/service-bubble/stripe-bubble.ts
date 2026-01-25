import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * StripeBubble - Stripe payment operations
 */
export class StripeBubble extends ServiceBubble<StripeParams, StripeResult> {
  bubbleName = 'stripe';
  type = 'service';
  alias = 'Stripe';
  credentialType = 'stripe_api_key';

  params = {
    apiKey: z.string().min(1),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const Stripe = await import('stripe');
    this.client = new Stripe.default(this.params.apiKey);
  }

  async createPaymentIntent(params: { amount: number; currency: string; customerId?: string; metadata?: any }): Promise<StripeResult> {
    try {
      const result = await this.client.paymentIntents.create({
        amount: params.amount,
        currency: params.currency,
        customer: params.customerId,
        metadata: params.metadata
      });
      return { success: true, paymentIntent: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async confirmPayment(params: { paymentIntentId: string; paymentMethod: string }): Promise<StripeResult> {
    try {
      const result = await this.client.paymentIntents.confirm(params.paymentIntentId, {
        payment_method: params.paymentMethod
      });
      return { success: true, paymentIntent: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async refundPayment(params: { paymentIntentId: string; amount?: number; reason?: string }): Promise<StripeResult> {
    try {
      const paymentIntent = await this.client.paymentIntents.retrieve(params.paymentIntentId);
      const result = await this.client.refunds.create({
        charge: paymentIntent.charges.data[0].id,
        amount: params.amount,
        reason: params.reason
      });
      return { success: true, refund: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createCustomer(params: { email: string; name?: string; metadata?: any }): Promise<StripeResult> {
    try {
      const result = await this.client.customers.create({
        email: params.email,
        name: params.name,
        metadata: params.metadata
      });
      return { success: true, customer: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getCustomer(params: { customerId: string }): Promise<StripeResult> {
    try {
      const result = await this.client.customers.retrieve(params.customerId);
      return { success: true, customer: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createSubscription(params: { customerId: string; priceId: string; metadata?: any }): Promise<StripeResult> {
    try {
      const result = await this.client.subscriptions.create({
        customer: params.customerId,
        items: [{ price: params.priceId }],
        metadata: params.metadata
      });
      return { success: true, subscription: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async cancelSubscription(params: { subscriptionId: string }): Promise<StripeResult> {
    try {
      const result = await this.client.subscriptions.cancel(params.subscriptionId);
      return { success: true, subscription: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async handleWebhook(params: { payload: any; signature: string; secret: string }): Promise<StripeResult> {
    try {
      const event = this.client.webhooks.constructEvent(params.payload, params.signature, params.secret);
      return { success: true, event };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface StripeParams {
  apiKey: string;
  timeout?: number;
}

export interface StripeResult {
  success: boolean;
  paymentIntent?: any;
  refund?: any;
  customer?: any;
  subscription?: any;
  event?: any;
  error?: string;
}
