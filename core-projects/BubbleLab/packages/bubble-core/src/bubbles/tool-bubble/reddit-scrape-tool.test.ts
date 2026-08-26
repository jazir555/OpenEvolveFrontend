/**
 * Reddit Scrape Tool - unit tests
 * Rewritten to exercise the real RedditScrapeTool API (performAction).
 * The external HTTP layer (HttpBubble) is mocked.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';

const { httpState } = vi.hoisted(() => ({
  httpState: {
    fail: false,
    data: {
      status: 200,
      statusText: 'OK',
      json: {
        data: {
          children: [
            {
              data: {
                title: 'Post 1',
                url: 'https://example.com/1',
                author: 'user1',
                score: 10,
                num_comments: 3,
                created_utc: 1700000000,
                permalink: '/r/test/comments/1',
                selftext: 'body one',
                subreddit: 'test',
                post_hint: null,
                is_self: true,
                thumbnail: 'thumb',
                domain: 'example.com',
                link_flair_text: null,
              },
            },
            {
              data: {
                title: 'Post 2',
                url: 'https://example.com/2',
                author: 'user2',
                score: 5,
                num_comments: 1,
                created_utc: 1700000100,
                permalink: '/r/test/comments/2',
                selftext: '',
                subreddit: 'test',
                post_hint: null,
                is_self: false,
                thumbnail: 'self',
                domain: 'example.com',
                link_flair_text: 'flair',
              },
            },
          ],
        },
      },
      body: '{}',
    },
  },
}));

vi.mock('../service-bubble/http.ts', () => {
  class MockHttpBubble {
    async action() {
      if (httpState.fail) {
        return { success: false, error: 'Network error' };
      }
      return { success: true, data: httpState.data };
    }
  }
  return { HttpBubble: MockHttpBubble };
});

import { RedditScrapeTool } from './reddit-scrape-tool';

describe('RedditScrapeTool', () => {
  afterEach(() => {
    httpState.fail = false;
  });

  it('constructs with a subreddit', () => {
    const tool = new RedditScrapeTool({ subreddit: 'programming' });
    expect(tool).toBeDefined();
    expect(tool.params.subreddit).toBe('programming');
  });

  it('strips the r/ prefix from subreddit', () => {
    const tool = new RedditScrapeTool({ subreddit: 'r/programming' });
    expect(tool.params.subreddit).toBe('programming');
  });

  it('scrapes posts successfully', async () => {
    const tool = new RedditScrapeTool({ subreddit: 'test', limit: 10 });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.posts).toHaveLength(2);
    expect(result.posts[0].title).toBe('Post 1');
    expect(result.metadata.subreddit).toBe('test');
    expect(result.metadata.actualCount).toBe(2);
  });

  it('handles fetch errors gracefully', async () => {
    httpState.fail = true;

    const tool = new RedditScrapeTool({ subreddit: 'test' });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Network error');
  });
});
