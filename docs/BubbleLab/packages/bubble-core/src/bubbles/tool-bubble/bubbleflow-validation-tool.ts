import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * BubbleflowValidationTool - Validate BubbleFlow definitions
 */
export class BubbleflowValidationTool extends ToolBubble<BubbleflowValidationParams, BubbleflowValidationResult> {
  bubbleName = 'bubbleflow-validation';
  type = 'tool';
  alias = 'bubbleflow-validation';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<BubbleflowValidationResult> {
    try {
      const result = await this.validate(input);
      return { success: true, validation: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async validate(params: { bubbleflow: any }): Promise<BubbleflowValidationResult> {
    try {
      const errors = [];
      const warnings = [];

      // Validate structure
      if (!params.bubbleflow.id) {
        errors.push({ field: 'id', message: 'Missing required field: id' });
      }

      if (!params.bubbleflow.bubbles || params.bubbleflow.bubbles.length === 0) {
        errors.push({ field: 'bubbles', message: 'At least one bubble is required' });
      }

      // Validate connections
      if (params.bubbleflow.connections) {
        params.bubbleflow.connections.forEach((conn, i) => {
          if (!conn.from || !conn.to) {
            errors.push({ field: `connections[${i}]`, message: 'Connection must have from and to' });
          }
        });
      }

      // Check for potential issues
      if (params.bubbleflow.bubbles && params.bubbleflow.bubbles.length > 50) {
        warnings.push({ message: 'Large number of bubbles may impact performance' });
      }

      return {
        success: true,
        valid: errors.length === 0,
        errors,
        warnings,
        score: Math.max(0, 100 - errors.length * 10 - warnings.length * 5)
      };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface BubbleflowValidationParams {
  timeout?: number;
}

export interface BubbleflowValidationResult {
  success: boolean;
  valid?: boolean;
  errors?: any[];
  warnings?: any[];
  score?: number;
  error?: string;
}
