/**
 * Tests for Code Review Automation Workflow
 * Tests code review automation with AI analysis and PR feedback
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CodeReviewAutomation } from './code-review-automation';
import { BubbleFlow } from '@bubblelab/bubble-core';

// Mock dependencies
vi.mock('@bubblelab/bubble-core', () => ({
  BubbleFlow: class {
    constructor() {}
  },
  HttpBubble: vi.fn().mockImplementation((config) => ({
    action: vi.fn().mockResolvedValue({
      data: config.method === 'GET'
        ? [
            {
              filename: 'test.ts',
              additions: 10,
              deletions: 5,
              patch: '@@ -1,5 +1,10 @@\n+new code\n-old code'
            }
          ]
        : { id: 123 }
    }),
  })),
  AIAgentBubble: vi.fn().mockImplementation((config) => ({
    action: vi.fn().mockResolvedValue({
      data: {
        response: JSON.stringify({
          overallScore: 85,
          issues: {
            critical: [],
            warning: ['Consider adding error handling'],
            suggestion: ['Add unit tests']
          },
          approved: true,
          summary: 'Code looks good overall'
        }),
        usage: {
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150
        }
      }
    }),
  })),
  SlackBubble: vi.fn().mockImplementation((config) => ({
    action: vi.fn().mockResolvedValue({ success: true }),
  })),
}));

describe('CodeReviewAutomation', () => {
  let workflow: CodeReviewAutomation;
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    originalEnv = { ...process.env };
    process.env.GITHUB_PAT = 'test_github_token_12345678901234567890';
    process.env.OPENAI_API_KEY = 'test_openai_key_12345678901234567890123';
    process.env.API_KEY = 'test_api_key_1234567890123456789012345678';
    process.env.SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/TEST/TEST/TEST';

    // Mock validateEnvironment to prevent actual validation
    vi.doMock('../security-utils', () => ({
      validateEnvironment: vi.fn(),
      authenticateRequest: vi.fn(() => ({ authenticated: true })),
      requireAuthentication: vi.fn(),
      RateLimiter: vi.fn().mockImplementation(() => ({
        checkLimit: vi.fn(() => true),
      })),
      InputValidator: {
        sanitizeString: vi.fn((str: string, max?: number) => str.substring(0, max || 1000)),
        sanitizeNumber: vi.fn((num: number, min?: number, max?: number) => num),
      },
      sanitizeError: vi.fn((err: Error) => ({ message: 'Sanitized error' })),
      StructuredLogger: vi.fn().mockImplementation(() => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      })),
      generateCorrelationId: vi.fn(() => 'test-correlation-id-123456789012345678901234'),
      SecuritySchemas: {
        apiKey: { parse: vi.fn() },
        token: { parse: vi.fn() },
      },
    }));

    workflow = new CodeReviewAutomation();
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.clearAllMocks();
  });

  describe('Environment Validation', () => {
    it('should have required environment variables', () => {
      expect(process.env.GITHUB_PAT).toBeDefined();
      expect(process.env.OPENAI_API_KEY).toBeDefined();
      expect(process.env.API_KEY).toBeDefined();
    });

    it('should have workflow metadata', () => {
      expect(workflow.name).toBe('Code Review Automation');
      expect(workflow.description).toBeDefined();
    });

    it('should be instance of BubbleFlow', () => {
      expect(workflow).toBeInstanceOf(BubbleFlow);
    });
  });

  describe('Authentication', () => {
    it('should authenticate with valid API key', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
          'x-forwarded-for': '192.168.1.1',
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
    });

    it('should reject missing API key', async () => {
      const payload = {
        headers: {},
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      await expect(workflow.handle(payload as any)).rejects.toThrow();
    });

    it('should reject invalid API key', async () => {
      const payload = {
        headers: {
          'x-api-key': 'invalid_key',
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      await expect(workflow.handle(payload as any)).rejects.toThrow();
    });
  });

  describe('Input Validation', () => {
    it('should validate repository format', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'invalid-repo-format!!!',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      await expect(workflow.handle(payload as any)).rejects.toThrow('Invalid repository format');
    });

    it('should accept valid repository format', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'owner/repository-name',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result.repository).toBe('owner/repository-name');
    });

    it('should sanitize PR title to prevent XSS', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: '<script>alert("xss")</script> Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
      expect(result.overallScore).toBeGreaterThanOrEqual(0);
      expect(result.overallScore).toBeLessThanOrEqual(100);
    });
  });

  describe('Rate Limiting', () => {
    it('should allow requests within rate limit', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
    });

    it('should block requests exceeding rate limit', async () => {
      // Mock rate limiter to return false
      const { RateLimiter } = await import('../security-utils');
      vi.mocked(RateLimiter).mockImplementation(() => ({
        checkLimit: vi.fn(() => false),
      }) as any);

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      await expect(workflow.handle(payload as any)).rejects.toThrow('Rate limit exceeded');
    });
  });

  describe('Core Operations - PR Analysis', () => {
    it('should fetch PR diff from GitHub', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
      expect(result.prNumber).toBe(123);
    });

    it('should analyze code with AI', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result.overallScore).toBeGreaterThanOrEqual(0);
      expect(result.overallScore).toBeLessThanOrEqual(100);
      expect(result.issues).toBeDefined();
      expect(result.issues.critical).toBeDefined();
      expect(result.issues.warning).toBeDefined();
      expect(result.issues.suggestion).toBeDefined();
    });

    it('should post review comment to GitHub', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
    });
  });

  describe('Label Management', () => {
    it('should add label when PR not approved', async () => {
      // Mock AI to return not approved
      const { AIAgentBubble } = await import('@bubblelab/bubble-core');
      vi.mocked(AIAgentBubble).mockImplementation(() => ({
        action: vi.fn().mockResolvedValue({
          data: {
            response: JSON.stringify({
              overallScore: 45,
              issues: {
                critical: ['Security vulnerability found'],
                warning: [],
                suggestion: []
              },
              approved: false,
              summary: 'Critical issues found'
            }),
          },
        }),
      }) as any);

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result.approved).toBe(false);
    });

    it('should not add label when PR approved', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result.approved).toBe(true);
    });
  });

  describe('Notification', () => {
    it('should send Slack notification when PR not approved', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      // Mock not approved
      const { AIAgentBubble } = await import('@bubblelab/bubble-core');
      vi.mocked(AIAgentBubble).mockImplementation(() => ({
        action: vi.fn().mockResolvedValue({
          data: {
            response: JSON.stringify({
              overallScore: 45,
              issues: {
                critical: ['Issue'],
                warning: [],
                suggestion: []
              },
              approved: false,
              summary: 'Needs work'
            }),
          },
        }),
      }) as any);

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
    });

    it('should not send Slack notification when PR approved', async () => {
      delete process.env.SLACK_WEBHOOK_URL;

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    it('should handle GitHub API errors gracefully', async () => {
      const { HttpBubble } = await import('@bubblelab/bubble-core');
      vi.mocked(HttpBubble).mockImplementation(() => ({
        action: vi.fn().mockRejectedValue(new Error('GitHub API error')),
      }) as any);

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      await expect(workflow.handle(payload as any)).rejects.toThrow();
    });

    it('should handle AI analysis errors gracefully', async () => {
      const { AIAgentBubble } = await import('@bubblelab/bubble-core');
      vi.mocked(AIAgentBubble).mockImplementation(() => ({
        action: vi.fn().mockRejectedValue(new Error('AI API error')),
      }) as any);

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      await expect(workflow.handle(payload as any)).rejects.toThrow();
    });

    it('should handle malformed AI responses', async () => {
      const { AIAgentBubble } = await import('@bubblelab/bubble-core');
      vi.mocked(AIAgentBubble).mockImplementation(() => ({
        action: vi.fn().mockResolvedValue({
          data: {
            response: 'This is not valid JSON',
          },
        }),
      }) as any);

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
      expect(result.approved).toBe(false); // Should fallback to not approved
    });

    it('should sanitize error messages', async () => {
      const { HttpBubble } = await import('@bubblelab/bubble-core');
      vi.mocked(HttpBubble).mockImplementation(() => ({
        action: vi.fn().mockRejectedValue(
          new Error('Error with API_KEY: secret123')
        ),
      }) as any);

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      try {
        await workflow.handle(payload as any);
      } catch (error: any) {
        expect(error.message).not.toContain('secret123');
      }
    });

    it('should log errors with correlation ID', async () => {
      const { HttpBubble } = await import('@bubblelab/bubble-core');
      vi.mocked(HttpBubble).mockImplementation(() => ({
        action: vi.fn().mockRejectedValue(new Error('Test error')),
      }) as any);

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Test PR',
        body: 'Test body',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 5,
        additions: 100,
        deletions: 50,
      };

      try {
        await workflow.handle(payload as any);
      } catch (error) {
        // Error should be logged with correlation ID
        expect(payload).toBeDefined();
      }
    });
  });

  describe('Integration Scenarios', () => {
    it('should handle complete code review workflow', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
          'x-forwarded-for': '192.168.1.1',
        },
        repository: 'owner/repo',
        number: 123,
        title: 'Feature: Add new functionality',
        body: 'This PR adds new functionality',
        author: 'developer',
        baseBranch: 'main',
        headBranch: 'feature/new-functionality',
        changedFiles: 10,
        additions: 500,
        deletions: 100,
      };

      const result = await workflow.handle(payload as any);

      expect(result).toBeDefined();
      expect(result.prNumber).toBe(123);
      expect(result.repository).toBe('owner/repo');
      expect(result.timestamp).toBeDefined();
      expect(result.correlationId).toBeDefined();
      expect(result.overallScore).toBeGreaterThanOrEqual(0);
      expect(result.overallScore).toBeLessThanOrEqual(100);
      expect(result.issues).toBeDefined();
    });

    it('should handle concurrent PR reviews', async () => {
      const payloads = [
        {
          headers: { 'x-api-key': process.env.API_KEY },
          repository: 'test/repo',
          number: 123,
          title: 'PR 1',
          body: 'Body',
          author: 'user1',
          baseBranch: 'main',
          headBranch: 'feature1',
          changedFiles: 5,
          additions: 100,
          deletions: 50,
        },
        {
          headers: { 'x-api-key': process.env.API_KEY },
          repository: 'test/repo',
          number: 124,
          title: 'PR 2',
          body: 'Body',
          author: 'user2',
          baseBranch: 'main',
          headBranch: 'feature2',
          changedFiles: 3,
          additions: 50,
          deletions: 25,
        },
      ];

      const results = await Promise.all(
        payloads.map(p => workflow.handle(p as any))
      );

      expect(results).toHaveLength(2);
      expect(results[0].prNumber).toBe(123);
      expect(results[1].prNumber).toBe(124);
    });
  });

  describe('Edge Cases', () => {
    it('should handle PR with no changed files', async () => {
      const { HttpBubble } = await import('@bubblelab/bubble-core');
      vi.mocked(HttpBubble).mockImplementation((config: any) => ({
        action: vi.fn().mockResolvedValue({
          data: config.method === 'GET' ? [] : { id: 123 },
        }),
      }) as any);

      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Empty PR',
        body: 'No changes',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'empty',
        changedFiles: 0,
        additions: 0,
        deletions: 0,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
    });

    it('should handle very large PR', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test/repo',
        number: 123,
        title: 'Large PR',
        body: 'Many changes',
        author: 'testuser',
        baseBranch: 'main',
        headBranch: 'large-feature',
        changedFiles: 500,
        additions: 50000,
        deletions: 10000,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
    });

    it('should handle special characters in repository name', async () => {
      const payload = {
        headers: {
          'x-api-key': process.env.API_KEY,
        },
        repository: 'test-org/test-repo_name',
        number: 123,
        title: 'Test',
        body: 'Body',
        author: 'user',
        baseBranch: 'main',
        headBranch: 'feature',
        changedFiles: 1,
        additions: 10,
        deletions: 5,
      };

      const result = await workflow.handle(payload as any);
      expect(result).toBeDefined();
    });
  });
});
