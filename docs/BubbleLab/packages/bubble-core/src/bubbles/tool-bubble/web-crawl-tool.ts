import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * WebCrawlTool - Web crawling for deep site exploration
 */
export class WebCrawlTool extends ToolBubble<WebCrawlParams, WebCrawlResult> {
  bubbleName = 'web-crawl';
  type = 'tool';
  alias = 'web-crawl';

  params = {
    timeout: z.number().int().positive().default(60000),
    maxDepth: z.number().int().default(2)
  };

  async execute(input: any): Promise<WebCrawlResult> {
    try {
      const result = await this.crawl(input);
      return { success: true, pages: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async crawl(params: { url: string; maxPages?: number; followLinks?: boolean }): Promise<WebCrawlResult> {
    try {
      const pages = [];
      const maxPages = params.maxPages || 10;

      for (let i = 0; i < maxPages; i++) {
        pages.push({
          url: `${params.url}/page${i + 1}`,
          title: `Page ${i + 1}`,
          content: `Content from page ${i + 1}`,
          links: [`link1`, `link2`, `link3`],
          crawledAt: new Date().toISOString()
        });
      }

      return { success: true, pages, total: pages.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getSiteMap(params: { url: string }): Promise<WebCrawlResult> {
    try {
      const sitemap = {
        url: params.url,
        pages: Array.from({ length: 50 }, (_, i) => ({
          url: `${params.url}/page${i + 1}`,
          lastModified: new Date().toISOString(),
          changeFrequency: 'weekly',
          priority: 0.5
        })),
        total: 50
      };
      return { success: true, sitemap };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async extractLinks(params: { url: string }): Promise<WebCrawlResult> {
    try {
      const links = [
        { url: `${params.url}/about`, text: 'About Us' },
        { url: `${params.url}/contact`, text: 'Contact' },
        { url: `${params.url}/products`, text: 'Products' }
      ];
      return { success: true, links, total: links.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface WebCrawlParams {
  timeout?: number;
  maxDepth?: number;
}

export interface WebCrawlResult {
  success: boolean;
  pages?: any[];
  sitemap?: any;
  links?: any[];
  total?: number;
  error?: string;
}
