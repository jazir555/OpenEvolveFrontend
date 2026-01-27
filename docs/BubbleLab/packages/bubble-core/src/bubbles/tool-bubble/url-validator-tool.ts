import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * URLValidatorTool - URL validation and normalization
 */
export class URLValidatorTool extends ToolBubble<URLValidatorParams, URLValidatorResult> {
  bubbleName = 'url-validator';
  type = 'tool';
  alias = 'url-validator';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<URLValidatorResult> {
    try {
      const result = await this.validate(input);
      return { success: true, valid: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async validate(params: { url: string }): Promise<URLValidatorResult> {
    try {
      const urlRegex = /^https?:\/\/.+/;
      const isValid = urlRegex.test(params.url);
      return { success: true, valid: isValid, url: params.url };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async normalize(params: { url: string }): Promise<URLValidatorResult> {
    try {
      let normalized = params.url.trim();
      if (!normalized.startsWith('http://') && !normalized.startsWith('https://')) {
        normalized = 'https://' + normalized;
      }
      return { success: true, normalized };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async check(params: { url: string }): Promise<URLValidatorResult> {
    try {
      // Placeholder for URL availability check
      return { success: true, reachable: true, statusCode: 200 };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface URLValidatorParams {
  timeout?: number;
}

export interface URLValidatorResult {
  success: boolean;
  valid?: boolean;
  url?: string;
  normalized?: string;
  reachable?: boolean;
  statusCode?: number;
  error?: string;
}
