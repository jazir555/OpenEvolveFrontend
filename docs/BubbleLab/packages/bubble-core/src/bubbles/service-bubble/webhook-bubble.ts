import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * WebhookBubble - Webhook receiving and processing operations
 */
export class WebhookBubble extends ServiceBubble<WebhookParams, WebhookResult> {
  bubbleName = 'webhook';
  type = 'service';
  alias = 'Webhook';
  credentialType = 'webhook_api_key';

  params = {
    secret: z.string().min(1),
    timeout: z.number().int().positive().default(30000)
  };

  private storage: Map<string, any> = new Map();

  async connect() {
    // Initialize webhook storage
    this.storage.clear();
  }

  async receiveWebhook(params: { payload: any; headers?: any; signature?: string }): Promise<WebhookResult> {
    try {
      const webhookId = `wh_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const webhook = {
        id: webhookId,
        payload: params.payload,
        headers: params.headers,
        signature: params.signature,
        receivedAt: new Date().toISOString(),
        processed: false
      };
      this.storage.set(webhookId, webhook);
      return { success: true, webhook };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async parsePayload(params: { payload: any; format: 'json' | 'form' | 'xml' }): Promise<WebhookResult> {
    try {
      let parsed = params.payload;
      if (params.format === 'json') {
        parsed = typeof params.payload === 'string' ? JSON.parse(params.payload) : params.payload;
      } else if (params.format === 'form') {
        const formData = new URLSearchParams(params.payload);
        parsed = Object.fromEntries(formData.entries());
      }
      return { success: true, parsed };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async validateSignature(params: { payload: any; signature: string; algorithm?: 'sha256' | 'sha1' | 'md5' }): Promise<WebhookResult> {
    try {
      const crypto = await import('crypto');
      const algo = params.algorithm || 'sha256';
      const hash = crypto.createHmac(algo, this.params.secret)
        .update(JSON.stringify(params.payload))
        .digest('hex');
      const isValid = hash === params.signature;
      return { success: true, valid: isValid, hash };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async dispatchEvent(params: { webhookId: string; eventType: string; data: any }): Promise<WebhookResult> {
    try {
      const webhook = this.storage.get(params.webhookId);
      if (!webhook) {
        return { success: false, error: 'Webhook not found' };
      }
      const event = {
        type: params.eventType,
        data: params.data,
        timestamp: new Date().toISOString(),
        webhookId: params.webhookId
      };
      return { success: true, event };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async replayWebhook(params: { webhookId: string }): Promise<WebhookResult> {
    try {
      const webhook = this.storage.get(params.webhookId);
      if (!webhook) {
        return { success: false, error: 'Webhook not found' };
      }
      const replayed = {
        ...webhook,
        replayedAt: new Date().toISOString(),
        replayCount: (webhook.replayCount || 0) + 1
      };
      this.storage.set(params.webhookId, replayed);
      return { success: true, webhook: replayed };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async listWebhooks(params?: { limit?: number }): Promise<WebhookResult> {
    try {
      const webhooks = Array.from(this.storage.values());
      const limited = params.limit ? webhooks.slice(0, params.limit) : webhooks;
      return { success: true, webhooks: limited };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deleteWebhook(params: { webhookId: string }): Promise<WebhookResult> {
    try {
      const deleted = this.storage.delete(params.webhookId);
      return { success: deleted, deleted: params.webhookId };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getStats(params?: {}): Promise<WebhookResult> {
    try {
      const webhooks = Array.from(this.storage.values());
      const stats = {
        total: webhooks.length,
        processed: webhooks.filter((w: any) => w.processed).length,
        pending: webhooks.filter((w: any) => !w.processed).length,
        byType: webhooks.reduce((acc: any, w: any) => {
          acc[w.type] = (acc[w.type] || 0) + 1;
          return acc;
        }, {})
      };
      return { success: true, stats };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface WebhookParams {
  secret: string;
  timeout?: number;
}

export interface WebhookResult {
  success: boolean;
  webhook?: any;
  webhooks?: any[];
  parsed?: any;
  valid?: boolean;
  hash?: string;
  event?: any;
  deleted?: string;
  stats?: any;
  error?: string;
}
