import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * GetBubbleDetailsTool - Retrieve detailed information about a bubble
 */
export class GetBubbleDetailsTool extends ToolBubble<GetBubbleDetailsParams, GetBubbleDetailsResult> {
  bubbleName = 'get-bubble-details';
  type = 'tool';
  alias = 'get-bubble-details';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<GetBubbleDetailsResult> {
    try {
      const result = await this.getDetails(input);
      return { success: true, details: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getDetails(params: { bubbleId: string }): Promise<GetBubbleDetailsResult> {
    try {
      const details = {
        id: params.bubbleId,
        name: 'Bubble Name',
        type: 'service',
        version: '1.0.0',
        description: 'Bubble description',
        operations: ['op1', 'op2', 'op3'],
        parameters: {
          apiKey: { type: 'string', required: true },
          baseUrl: { type: 'string', required: true }
        },
        createdAt: '2025-01-17T00:00:00Z',
        updatedAt: '2025-01-17T00:00:00Z'
      };
      return { success: true, details };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface GetBubbleDetailsParams {
  timeout?: number;
}

export interface GetBubbleDetailsResult {
  success: boolean;
  details?: any;
  error?: string;
}
