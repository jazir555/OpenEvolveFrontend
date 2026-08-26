/**
 * Web Scrape Tool - unit tests
 * Rewritten to exercise the real WebScrapeTool API (performAction).
 * The external FirecrawlBubble is mocked.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';

const { fcState } = vi.hoisted(() => ({ fcState: { fail: false } }));

vi.mock('../service-bubble/firecrawl.ts', () => {
  class MockFirecrawlBubble {
    async action() {
      if (fcState.fail) {
        throw new Error('Forbidden');
      }
      return {
        success: true,
        data: {
          markdown: '# Title\n\nSome scraped content here.',
          html: '<h1>Title</h1><p>Some scraped content here.</p>',
          metadata: { title: 'Title', statusCode: 200 },
        },
      };
    }
  }
  return { FirecrawlBubble: MockFirecrawlBubble };
});

import { WebScrapeTool } from './web-scrape-tool';

describe('WebScrapeTool', () => {
  afterEach(() => {
    fcState.fail = false;
  });

  it('constructs with a url', () => {
    const tool = new WebScrapeTool({ url: 'https://example.com' });
    expect(tool).toBeDefined();
  });

  it('scrapes markdown content', async () => {
    const tool = new WebScrapeTool({ url: 'https://example.com' });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.content).toContain('Some scraped content');
    expect(result.title).toBe('Title');
    expect(result.format).toBe('markdown');
    expect(result.creditsUsed).toBe(1);
    expect(result.url).toBe('https://example.com');
  });

  it('handles scrape errors', async () => {
    fcState.fail = true;
    const tool = new WebScrapeTool({ url: 'https://example.com' });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Forbidden');
  });
});
