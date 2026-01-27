import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * NotionBubble - Notion workspace and database operations
 */
export class NotionBubble extends ServiceBubble<NotionParams, NotionResult> {
  bubbleName = 'notion';
  type = 'service';
  alias = 'Notion';
  credentialType = 'notion_api_key';

  params = {
    token: z.string().min(1),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { Client } = await import('@notionhq/client');
    this.client = new Client({ auth: this.params.token });
  }

  async createPage(params: { parentId: string; title: string; children?: any[] }): Promise<NotionResult> {
    try {
      const result = await this.client.pages.create({
        parent: { page_id: params.parentId },
        properties: {
          title: {
            title: [{ text: { content: params.title } }]
          }
        },
        children: params.children
      });
      return { success: true, page: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getPage(params: { pageId: string }): Promise<NotionResult> {
    try {
      const result = await this.client.pages.retrieve({ page_id: params.pageId });
      return { success: true, page: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async updatePage(params: { pageId: string; properties: any }): Promise<NotionResult> {
    try {
      const result = await this.client.pages.update({
        page_id: params.pageId,
        properties: params.properties
      });
      return { success: true, page: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deletePage(params: { pageId: string }): Promise<NotionResult> {
    try {
      const result = await this.client.pages.update({
        page_id: params.pageId,
        archived: true
      });
      return { success: true, page: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async queryDatabase(params: { databaseId: string; filter?: any; sorts?: any }): Promise<NotionResult> {
    try {
      const result = await this.client.databases.query({
        database_id: params.databaseId,
        filter: params.filter,
        sorts: params.sorts
      });
      return { success: true, results: result.results };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createDatabase(params: { parentId: string; title: string; schema: any }): Promise<NotionResult> {
    try {
      const result = await this.client.databases.create({
        parent: { page_id: params.parentId },
        title: [{ type: 'text', text: { content: params.title } }],
        properties: params.schema
      });
      return { success: true, database: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async appendBlock(params: { blockId: string; children: any[] }): Promise<NotionResult> {
    try {
      const result = await this.client.blocks.children.append({
        block_id: params.blockId,
        children: params.children
      });
      return { success: true, block: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async searchPages(params: { query: string; filter?: any }): Promise<NotionResult> {
    try {
      const result = await this.client.search({
        query: params.query,
        filter: params.filter
      });
      return { success: true, results: result.results };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface NotionParams {
  token: string;
  timeout?: number;
}

export interface NotionResult {
  success: boolean;
  page?: any;
  pages?: any[];
  results?: any[];
  database?: any;
  block?: any;
  error?: string;
}
