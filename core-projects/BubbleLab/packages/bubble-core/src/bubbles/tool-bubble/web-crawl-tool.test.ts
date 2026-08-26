/**
 * Web Crawl Tool - unit tests
 * Rewritten to exercise the real WebCrawlTool API (performAction).
 * The external FirecrawlBubble is mocked.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';

const { fcState } = vi.hoisted(() => ({ fcState: { fail: false } }));

vi.mock('../service-bubble/firecrawl.ts', () => {
  class MockFirecrawlBubble {
    async action() {
      if (fcState.fail) {
        return { success: false, error: 'Unauthorized' };
      }
      return {
        success: true,
        data: {
          completed: true,
          data: [
            {
              markdown: 'Page one content',
              metadata: {
                sourceURL: 'https://example.com/page1',
                title: 'Page 1',
                depth: 1,
              },
            },
            {
              markdown: 'Page two content',
              metadata: {
                sourceURL: 'https://example.com/page2',
                title: 'Page 2',
                depth: 2,
              },
            },
          ],
        },
      };
    }
  }
  return { FirecrawlBubble: MockFirecrawlBubble };
});

import { WebCrawlTool } from './web-crawl-tool';

describe('WebCrawlTool', () => {
  afterEach(() => {
    fcState.fail = false;
  });

  it('constructs with a url', () => {
    const tool = new WebCrawlTool({ url: 'https://example.com' });
    expect(tool).toBeDefined();
  });

  it('crawls multiple pages', async () => {
    const tool = new WebCrawlTool({ url: 'https://example.com' });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.pages).toHaveLength(2);
    expect(result.totalPages).toBe(2);
    expect(result.pages[0].url).toBe('https://example.com/page1');
    expect(result.creditsUsed).toBe(2);
  });

  it('handles crawl errors', async () => {
    fcState.fail = true;
    const tool = new WebCrawlTool({ url: 'https://example.com' });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Unauthorized');
  });
});
