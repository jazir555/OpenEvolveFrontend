/**
 * YouTube Tool - unit tests
 * Rewritten to exercise the real YouTubeTool API (performAction).
 * The external ApifyBubble is mocked; the no-credentials path is also covered.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { CredentialType } from '@bubblelab/shared-schemas';

const { apifyState } = vi.hoisted(() => ({ apifyState: { fail: false } }));

vi.mock('../service-bubble/apify/apify.ts', () => {
  class MockApifyBubble {
    async action() {
      if (apifyState.fail) {
        return { data: { success: false, error: 'Actor failed' } };
      }
      return {
        data: {
          success: true,
          items: [
            {
              title: 'My Video',
              id: 'vid1',
              url: 'https://www.youtube.com/watch?v=vid1',
            },
          ],
        },
      };
    }
  }
  return { ApifyBubble: MockApifyBubble };
});

import { YouTubeTool } from './youtube-tool';

describe('YouTubeTool', () => {
  afterEach(() => {
    apifyState.fail = false;
  });

  it('requires APIFY credentials', async () => {
    const tool = new YouTubeTool({ operation: 'searchVideos', searchQueries: ['ai'] });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('authentication');
  });

  it('requires search queries or urls for searchVideos', async () => {
    const tool = new YouTubeTool({
      operation: 'searchVideos',
      credentials: { [CredentialType.APIFY_CRED]: 'key' },
    });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('searchVideos requires');
  });

  it('searches videos when credentials are provided', async () => {
    const tool = new YouTubeTool({
      operation: 'searchVideos',
      searchQueries: ['ai'],
      credentials: { [CredentialType.APIFY_CRED]: 'key' },
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.videos).toHaveLength(1);
    expect(result.videos![0].title).toBe('My Video');
    expect(result.totalResults).toBe(1);
  });

  it('handles actor failures gracefully', async () => {
    apifyState.fail = true;
    const tool = new YouTubeTool({
      operation: 'searchVideos',
      searchQueries: ['ai'],
      credentials: { [CredentialType.APIFY_CRED]: 'key' },
    });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Actor failed');
  });
});
