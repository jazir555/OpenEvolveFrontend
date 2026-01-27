import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * ChartJSTool - Chart generation and visualization using Chart.js
 */
export class ChartJSTool extends ToolBubble<ChartJSParams, ChartJSResult> {
  bubbleName = 'chart-js';
  type = 'tool';
  alias = 'chart-js';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<ChartJSResult> {
    try {
      const result = await this.generate(input);
      return { success: true, chart: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async generate(params: { type: string; data: any; options?: any }): Promise<ChartJSResult> {
    try {
      const chart = {
        type: params.type,
        data: params.data,
        options: params.options || {},
        generatedAt: new Date().toISOString()
      };
      return { success: true, chart };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface ChartJSParams {
  timeout?: number;
}

export interface ChartJSResult {
  success: boolean;
  chart?: any;
  error?: string;
}
