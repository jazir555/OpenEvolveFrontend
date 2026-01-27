import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * SendGridBubble - SendGrid email operations
 */
export class SendGridBubble extends ServiceBubble<SendGridParams, SendGridResult> {
  bubbleName = 'sendgrid';
  type = 'service';
  alias = 'SendGrid';
  credentialType = 'sendgrid_api_key';

  params = {
    apiKey: z.string().min(1),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const sgMail = await import('@sendgrid/mail');
    sgMail.default.setApiKey(this.params.apiKey);
    this.client = sgMail.default;
  }

  async sendEmail(params: { to: string; from: string; subject: string; text?: string; html?: string }): Promise<SendGridResult> {
    try {
      const result = await this.client.send({
        to: params.to,
        from: params.from,
        subject: params.subject,
        text: params.text,
        html: params.html
      });
      return { success: true, messageId: result[0]?.headers['x-message-id'] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async sendBulkEmails(params: { messages: any[] }): Promise<SendGridResult> {
    try {
      const result = await this.client.send(params.messages);
      return { success: true, messageIds: result.map((r: any) => r.headers['x-message-id']) };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async sendTemplate(params: { to: string; from: string; templateId: string; dynamicData?: any }): Promise<SendGridResult> {
    try {
      const result = await this.client.send({
        to: params.to,
        from: params.from,
        templateId: params.templateId,
        dynamicTemplateData: params.dynamicData || {}
      });
      return { success: true, messageId: result[0]?.headers['x-message-id'] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async addContact(params: { listId: string; email: string; firstName?: string; lastName?: string }): Promise<SendGridResult> {
    try {
      const result = await this.client.client.request({
        method: 'PUT',
        url: `/v3/marketing/contacts`,
        body: {
          list_ids: [params.listId],
          contacts: [{
            email: params.email,
            first_name: params.firstName,
            last_name: params.lastName
          }]
        }
      });
      return { success: true, contact: result.body };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getContact(params: { email: string }): Promise<SendGridResult> {
    try {
      const result = await this.client.client.request({
        method: 'GET',
        url: `/v3/marketing/contacts/search`,
        qs: { query: `email LIKE '${params.email}'` }
      });
      return { success: true, contact: result.body };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deleteContact(params: { email: string }): Promise<SendGridResult> {
    try {
      const result = await this.client.client.request({
        method: 'DELETE',
        url: '/v3/marketing/contacts',
        body: { emails: [params.email] }
      });
      return { success: true, deleted: params.email };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createList(params: { name: string }): Promise<SendGridResult> {
    try {
      const result = await this.client.client.request({
        method: 'POST',
        url: '/v3/marketing/lists',
        body: { name: params.name }
      });
      return { success: true, list: result.body };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async addToList(params: { listId: string; contacts: any[] }): Promise<SendGridResult> {
    try {
      const result = await this.client.client.request({
        method: 'PUT',
        url: '/v3/marketing/contacts',
        body: {
          list_ids: [params.listId],
          contacts: params.contacts
        }
      });
      return { success: true, result: result.body };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface SendGridParams {
  apiKey: string;
  timeout?: number;
}

export interface SendGridResult {
  success: boolean;
  messageId?: string;
  messageIds?: string[];
  contact?: any;
  list?: any;
  result?: any;
  deleted?: string;
  error?: string;
}
