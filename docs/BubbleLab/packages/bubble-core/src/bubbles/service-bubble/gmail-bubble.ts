import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * GmailBubble - Gmail email operations
 */
export class GmailBubble extends ServiceBubble<GmailParams, GmailResult> {
  bubbleName = 'gmail';
  type = 'service';
  alias = 'Gmail';
  credentialType = 'gmail_api_key';

  params = {
    credentials: z.any(),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { gmail } = await import('@googleapis/gmail');
    const auth = await import('google-auth-library');
    const authClient = auth.JWT.fromJSON(this.params.credentials);
    this.client = gmail({ version: 'v1', auth: authClient });
  }

  async sendEmail(params: { to: string[]; subject: string; body: string; cc?: string[]; bcc?: string[] }): Promise<GmailResult> {
    try {
      const email = [
        `To: ${params.to.join(', ')}`,
        params.cc ? `Cc: ${params.cc.join(', ')}` : '',
        params.bcc ? `Bcc: ${params.bcc.join(', ')}` : '',
        `Subject: ${params.subject}`,
        '',
        params.body
      ].filter(Boolean).join('\r\n');

      const encodedEmail = Buffer.from(email).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

      const result = await this.client.users.messages.send({
        userId: 'me',
        requestBody: { raw: encodedEmail }
      });
      return { success: true, message: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async listMessages(params: { maxResults?: number; labelIds?: string[]; query?: string }): Promise<GmailResult> {
    try {
      const result = await this.client.users.messages.list({
        userId: 'me',
        maxResults: params.maxResults || 10,
        labelIds: params.labelIds,
        q: params.query
      });
      return { success: true, messages: result.data.messages || [] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getMessage(params: { id: string; format?: string }): Promise<GmailResult> {
    try {
      const result = await this.client.users.messages.get({
        userId: 'me',
        id: params.id,
        format: params.format || 'full'
      });
      return { success: true, message: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async searchMessages(params: { query: string; maxResults?: number }): Promise<GmailResult> {
    try {
      const result = await this.client.users.messages.list({
        userId: 'me',
        q: params.query,
        maxResults: params.maxResults || 10
      });
      return { success: true, messages: result.data.messages || [] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async modifyLabels(params: { id: string; addLabelIds?: string[]; removeLabelIds?: string[] }): Promise<GmailResult> {
    try {
      const result = await this.client.users.messages.modify({
        userId: 'me',
        id: params.id,
        requestBody: {
          addLabelIds: params.addLabelIds || [],
          removeLabelIds: params.removeLabelIds || []
        }
      });
      return { success: true, message: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async listLabels(params?: {}): Promise<GmailResult> {
    try {
      const result = await this.client.users.labels.list({ userId: 'me' });
      return { success: true, labels: result.data.labels || [] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createLabel(params: { name: string; labelListVisibility?: string; messageListVisibility?: string }): Promise<GmailResult> {
    try {
      const result = await this.client.users.labels.create({
        userId: 'me',
        requestBody: {
          name: params.name,
          labelListVisibility: params.labelListVisibility || 'labelShow',
          messageListVisibility: params.messageListVisibility || 'show'
        }
      });
      return { success: true, label: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deleteMessage(params: { id: string }): Promise<GmailResult> {
    try {
      await this.client.users.messages.delete({
        userId: 'me',
        id: params.id
      });
      return { success: true, deleted: params.id };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface GmailParams {
  credentials: any;
  timeout?: number;
}

export interface GmailResult {
  success: boolean;
  message?: any;
  messages?: any[];
  labels?: any[];
  label?: any;
  deleted?: string;
  error?: string;
}
