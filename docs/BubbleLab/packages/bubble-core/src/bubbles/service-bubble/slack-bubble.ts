import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * SlackBubble - Slack messaging and channel operations
 */
export class SlackBubble extends ServiceBubble<SlackParams, SlackResult> {
  bubbleName = 'slack';
  type = 'service';
  alias = 'Slack';
  credentialType = 'slack_api_key';

  params = {
    token: z.string().min(1),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { WebClient } = await import('@slack/web-api');
    this.client = new WebClient(this.params.token);
  }

  async sendMessage(params: { channel: string; text: string; blocks?: any[] }): Promise<SlackResult> {
    try {
      const result = await this.client.chat.postMessage({
        channel: params.channel,
        text: params.text,
        blocks: params.blocks
      });
      return { success: true, message: result.message };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async listChannels(params: { types?: string; limit?: number }): Promise<SlackResult> {
    try {
      const result = await this.client.conversations.list({
        types: params.types || 'public_channel,private_channel',
        limit: params.limit || 100
      });
      return { success: true, channels: result.channels };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async addReaction(params: { channel: string; timestamp: string; name: string }): Promise<SlackResult> {
    try {
      await this.client.reactions.add({
        channel: params.channel,
        timestamp: params.timestamp,
        name: params.name
      });
      return { success: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async uploadFile(params: { channels: string[]; file: any; filename?: string; title?: string }): Promise<SlackResult> {
    try {
      const result = await this.client.files.uploadV2({
        channels: params.channels,
        file: params.file,
        filename: params.filename,
        title: params.title
      });
      return { success: true, file: result.file };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async scheduleMessage(params: { channel: string; text: string; postAt: number }): Promise<SlackResult> {
    try {
      const result = await this.client.chat.scheduleMessage({
        channel: params.channel,
        text: params.text,
        post_at: params.postAt
      });
      return { success: true, scheduledMessageId: result.scheduled_message_id };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async listUsers(params: { limit?: number; cursor?: string }): Promise<SlackResult> {
    try {
      const result = await this.client.users.list({
        limit: params.limit || 100,
        cursor: params.cursor
      });
      return { success: true, members: result.members, nextCursor: result.response_metadata?.next_cursor };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createChannel(params: { name: string; isPrivate?: boolean }): Promise<SlackResult> {
    try {
      const result = params.isPrivate
        ? await this.client.conversations.create({ name: params.name, is_private: true })
        : await this.client.conversations.create({ name: params.name });
      return { success: true, channel: result.channel };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async inviteToChannel(params: { channel: string; users: string[] }): Promise<SlackResult> {
    try {
      const result = await this.client.conversations.invite({
        channel: params.channel,
        users: params.users.join(',')
      });
      return { success: true, channel: result.channel };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface SlackParams {
  token: string;
  timeout?: number;
}

export interface SlackResult {
  success: boolean;
  message?: any;
  channels?: any[];
  file?: any;
  scheduledMessageId?: string;
  members?: any[];
  nextCursor?: string;
  channel?: any;
  error?: string;
}
