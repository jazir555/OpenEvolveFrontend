import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { GithubBubble } from './github-bubble';
import { CredentialType } from '@bubblelab/shared-schemas';

const creds = { [CredentialType.GITHUB_CRED]: 'test-github-token' };

describe('GithubBubble', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ id: 1, name: 'repo', success: true }),
      }))
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('should expose correct static metadata', () => {
    expect(GithubBubble.bubbleName).toBe('github');
    expect(GithubBubble.service).toBe('github');
    expect(GithubBubble.type).toBe('service');
    expect(GithubBubble.authType).toBe('token');
    expect(GithubBubble.schema).toBeDefined();
  });

  it('should accept valid params via the schema', () => {
    const result = GithubBubble.schema.safeParse({
      operation: 'getRepository',
      owner: 'octocat',
      repo: 'hello-world',
      credentials: creds,
    });
    expect(result.success).toBe(true);
  });

  it('should reject params missing required fields via the schema', () => {
    const result = GithubBubble.schema.safeParse({
      operation: 'createIssue',
      credentials: creds,
    });
    expect(result.success).toBe(false);
  });

  it('should NOT throw on invalid params (constructor is resilient)', () => {
    expect(() => new GithubBubble({} as any)).not.toThrow();
  });

  it('should return a controlled error from action() for invalid params', async () => {
    const bubble = new GithubBubble({} as any);
    const result = await bubble.action();
    expect(result.success).toBe(false);
    expect(result.error).toContain('Input Schema validation failed');
  });

  it('should run a valid operation end-to-end with mocked fetch', async () => {
    const bubble = new GithubBubble({
      operation: 'getRepository',
      owner: 'octocat',
      repo: 'hello-world',
      credentials: creds,
    });
    const result = await bubble.action();
    expect(result).toBeDefined();
    expect(typeof result.success).toBe('boolean');
    expect(typeof result.error).toBe('string');
  });
});
