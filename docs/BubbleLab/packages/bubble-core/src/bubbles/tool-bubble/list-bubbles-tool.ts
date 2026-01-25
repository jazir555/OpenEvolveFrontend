import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * ListBubblesTool - List all available bubbles in the system
 */
export class ListBubblesTool extends ToolBubble<ListBubblesParams, ListBubblesResult> {
  bubbleName = 'list-bubbles';
  type = 'tool';
  alias = 'list-bubbles';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<ListBubblesResult> {
    try {
      const result = await this.list(input);
      return { success: true, bubbles: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async list(params: { type?: 'service' | 'tool' | 'workflow' }): Promise<ListBubblesResult> {
    try {
      const bubbles = [
        { id: 'elasticsearch', name: 'Elasticsearch', type: 'service' },
        { id: 'redis', name: 'Redis', type: 'service' },
        { id: 'slack', name: 'Slack', type: 'service' },
        { id: 'github', name: 'GitHub', type: 'service' },
        { id: 'web-search', name: 'Web Search', type: 'tool' },
        { id: 'csv-processor', name: 'CSV Processor', type: 'tool' },
        { id: 'database-analyzer', name: 'Database Analyzer', type: 'workflow' }
      ];

      const filtered = params.type
        ? bubbles.filter(b => b.type === params.type)
        : bubbles;

      return { success: true, bubbles: filtered, total: filtered.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface ListBubblesParams {
  timeout?: number;
}

export interface ListBubblesResult {
  success: boolean;
  bubbles?: any[];
  total?: number;
  error?: string;
}
