import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * TwilioBubble - Twilio communication operations
 */
export class TwilioBubble extends ServiceBubble<TwilioParams, TwilioResult> {
  bubbleName = 'twilio';
  type = 'service';
  alias = 'Twilio';
  credentialType = 'twilio_api_key';

  params = {
    accountSid: z.string().min(1),
    authToken: z.string().min(1),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { Twilio } = await import('twilio');
    this.client = new Twilio(this.params.accountSid, this.params.authToken);
  }

  async sendSMS(params: { to: string; from: string; body: string }): Promise<TwilioResult> {
    try {
      const result = await this.client.messages.create({
        to: params.to,
        from: params.from,
        body: params.body
      });
      return { success: true, message: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async makeCall(params: { to: string; from: string; url: string; method?: string }): Promise<TwilioResult> {
    try {
      const result = await this.client.calls.create({
        to: params.to,
        from: params.from,
        url: params.url,
        method: params.method || 'POST'
      });
      return { success: true, call: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async sendWhatsApp(params: { to: string; from: string; body: string }): Promise<TwilioResult> {
    try {
      const result = await this.client.messages.create({
        to: `whatsapp:${params.to}`,
        from: `whatsapp:${params.from}`,
        body: params.body
      });
      return { success: true, message: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async lookupNumber(params: { phoneNumber: string }): Promise<TwilioResult> {
    try {
      const result = await this.client.lookups.v1.phoneNumbers(params.phoneNumber).fetch();
      return { success: true, numberInfo: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createMessage(params: { to: string; from: string; body: string; mediaUrl?: string[] }): Promise<TwilioResult> {
    try {
      const result = await this.client.messages.create({
        to: params.to,
        from: params.from,
        body: params.body,
        mediaUrl: params.mediaUrl
      });
      return { success: true, message: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getMessage(params: { messageSid: string }): Promise<TwilioResult> {
    try {
      const result = await this.client.messages(params.messageSid).fetch();
      return { success: true, message: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getMedia(params: { messageSid: string; mediaSid: string }): Promise<TwilioResult> {
    try {
      const result = await this.client.messages(params.messageSid).media(params.mediaSid).fetch();
      return { success: true, media: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async validateNumber(params: { phoneNumber: string }): Promise<TwilioResult> {
    try {
      const result = await this.client.lookups.v1.phoneNumbers(params.phoneNumber)
        .fetch({ type: ['carrier'] });
      return { success: true, validation: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface TwilioParams {
  accountSid: string;
  authToken: string;
  timeout?: number;
}

export interface TwilioResult {
  success: boolean;
  message?: any;
  call?: any;
  numberInfo?: any;
  media?: any;
  validation?: any;
  error?: string;
}
