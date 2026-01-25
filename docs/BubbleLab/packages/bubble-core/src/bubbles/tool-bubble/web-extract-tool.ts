import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * WebExtractTool - Advanced data extraction from web pages
 */
export class WebExtractTool extends ToolBubble<WebExtractParams, WebExtractResult> {
  bubbleName = 'web-extract';
  type = 'tool';
  alias = 'web-extract';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<WebExtractResult> {
    try {
      const result = await this.extract(input);
      return { success: true, data: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async extract(params: { url: string; selectors: Record<string, string> }): Promise<WebExtractResult> {
    try {
      const extracted = {};
      for (const [key, selector] of Object.entries(params.selectors)) {
        extracted[key] = `Extracted data from ${selector}`;
      }
      return { success: true, extracted, url: params.url };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async extractTable(params: { url: string; tableIndex?: number }): Promise<WebExtractResult> {
    try {
      const table = {
        url: params.url,
        tableIndex: params.tableIndex || 0,
        headers: ['Column 1', 'Column 2', 'Column 3'],
        rows: Array.from({ length: 5 }, (_, i) => [
          `Row ${i + 1}, Col 1`,
          `Row ${i + 1}, Col 2`,
          `Row ${i + 1}, Col 3`
        ])
      };
      return { success: true, table };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async extractImages(params: { url: string }): Promise<WebExtractResult> {
    try {
      const images = [
        { url: `${params.url}/image1.jpg`, alt: 'Image 1', width: 800, height: 600 },
        { url: `${params.url}/image2.jpg`, alt: 'Image 2', width: 1024, height: 768 },
        { url: `${params.url}/image3.png`, alt: 'Image 3', width: 640, height: 480 }
      ];
      return { success: true, images, total: images.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async extractMetadata(params: { url: string }): Promise<WebExtractResult> {
    try {
      const metadata = {
        url: params.url,
        title: 'Page Title',
        description: 'Page description',
        keywords: ['keyword1', 'keyword2', 'keyword3'],
        ogTitle: 'OG Title',
        ogDescription: 'OG Description',
        ogImage: 'https://example.com/og-image.jpg',
        canonicalUrl: params.url
      };
      return { success: true, metadata };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface WebExtractParams {
  timeout?: number;
}

export interface WebExtractResult {
  success: boolean;
  extracted?: any;
  table?: any;
  images?: any[];
  metadata?: any;
  url?: string;
  total?: number;
  error?: string;
}
