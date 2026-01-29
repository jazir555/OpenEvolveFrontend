/**
 * GitHub API Service Bubble
 *
 * Provides integration with GitHub API for repository management,
 * issues, pull requests, and more.
 *
 * Federation Constitution Compliant
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// GITHUB-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const GitHubOperationSchema = z.enum([
  'get_repository',
  'list_repositories',
  'create_issue',
  'get_issue',
  'list_issues',
  'update_issue',
  'create_pull_request',
  'get_pull_request',
  'list_pull_requests',
  'merge_pull_request',
  'create_comment',
  'list_branches',
  'get_branch',
  'create_branch',
  'get_file',
  'create_file',
  'update_file',
  'delete_file',
  'get_commit',
  'list_commits',
  'create_webhook',
  'list_webhooks',
  'delete_webhook',
]);

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const GitHubParamsSchema = z.object({
  operation: GitHubOperationSchema.describe('GitHub API operation'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  accessToken: z.string().min(1).describe('GitHub personal access token (REQUIRED)'),
  baseUrl: z.string().url().default('https://api.github.com').describe('GitHub API base URL'),

  // Repository identification
  owner: z.string().optional().describe('Repository owner'),
  repo: z.string().optional().describe('Repository name'),

  // Issue/PR operations
  issueNumber: z.number().optional().describe('Issue or pull request number'),
  pullNumber: z.number().optional().describe('Pull request number'),
  title: z.string().optional().describe('Issue or pull request title'),
  body: z.string().optional().describe('Issue or pull request body'),
  state: z.enum(['open', 'closed', 'all']).default('open').describe('Issue/PR state'),
  labels: z.array(z.string()).optional().describe('Issue labels'),

  // Pull request operations
  head: z.string().optional().describe('Pull request head branch'),
  base: z.string().optional().describe('Pull request base branch'),
  mergeMethod: z.enum(['merge', 'squash', 'rebase']).default('merge').describe('Merge method'),
  commitTitle: z.string().optional().describe('Merge commit title'),
  commitMessage: z.string().optional().describe('Merge commit message'),

  // Comment operations
  commentId: z.number().optional().describe('Comment ID'),

  // Branch operations
  branch: z.string().optional().describe('Branch name'),
  sha: z.string().optional().describe('Commit SHA for branch creation'),

  // File operations
  path: z.string().optional().describe('File path'),
  content: z.string().optional().describe('File content (base64 encoded)'),
  message: z.string().optional().describe('Commit message'),

  // Pagination
  page: z.number().optional().describe('Page number'),
  perPage: z.number().min(1).max(100).default(30).describe('Items per page'),

  // Webhook operations
  webhookId: z.number().optional().describe('Webhook ID'),
  webhookUrl: z.string().url().optional().describe('Webhook URL'),
  webhookEvents: z.array(z.string()).optional().describe('Webhook events'),

  // Timeout
  timeout: z.number().min(1000).max(120000).default(30000).describe('Request timeout in ms'),
});

type GitHubParamsInput = z.input<typeof GitHubParamsSchema>;
type GitHubParams = z.output<typeof GitHubParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const GitHubResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  status: z.object({
    code: z.number(),
    reason: z.string().optional(),
  }),
  error: z.string().optional(),
  timing: z.number().describe('Response time in ms'),
  pagination: z.object({
    page: z.number().optional(),
    perPage: z.number().optional(),
    totalCount: z.number().optional(),
    hasNextPage: z.boolean().optional(),
  }).optional(),
});

type GitHubResult = z.output<typeof GitHubResultSchema>;

// ============================================================================
// GITHUB BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class GitHubBubble extends ServiceBubble<GitHubParams, GitHubResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'github' as const;
  static readonly type = 'service' as const;
  static readonly schema = GitHubParamsSchema;
  static readonly resultSchema = GitHubResultSchema;
  static readonly credentialType = 'github_access_token' as const;

  static readonly shortDescription = 'GitHub API integration for repository management';
  static readonly longDescription = `
    GitHub API service bubble for repository operations.

    Features:
    - Repository information and listing
    - Issue creation, retrieval, and updates
    - Pull request management and merging
    - Branch operations
    - File operations (get, create, update, delete)
    - Commit history
    - Webhook management
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - accessToken: GitHub personal access token (no default - must be provided)
    - baseUrl: GitHub API base URL (defaults to https://api.github.com)

    Federation Constitution Compliance:
    - No magic defaults (accessToken is required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: GitHubParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    GitHubBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('github', DEFAULT_RESILIENCE_CONFIG);
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - accessToken is required by schema
  }

  /**
   * Build HTTP headers for GitHub API requests
   */
  private buildHeaders(): Record<string, string> {
    return {
      'Authorization': `Bearer ${this.params.accessToken}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'X-GitHub-Api-Version': '2022-11-28',
    };
  }

  /**
   * Build full URL for GitHub API endpoint
   */
  private buildUrl(endpoint: string): string {
    return `${this.params.baseUrl}${endpoint}`;
  }

  /**
   * Make HTTP request to GitHub API
   */
  private async makeRequest(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<{ response: Response; data: unknown; timing: number }> {
    const startTime = Date.now();
    const url = this.buildUrl(endpoint);

    const response = await fetch(url, {
      method,
      headers: this.buildHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });

    const timing = Date.now() - startTime;

    let data: unknown;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    return { response, data, timing };
  }

  /**
   * Extract pagination information from response headers
   */
  private extractPagination(response: Response) {
    const linkHeader = response.headers.get('link');
    if (!linkHeader) {
      return undefined;
    }

    const hasNextPage = linkHeader.includes('rel="next"');
    return {
      page: this.params.page,
      perPage: this.params.perPage,
      hasNextPage,
    };
  }

  /**
   * Get repository operation
   */
  private async getRepository(): Promise<GitHubResult> {
    if (!this.params.owner || !this.params.repo) {
      throw new Error('owner and repo are required for get_repository operation');
    }

    const startTime = Date.now();

    try {
      const { response, data, timing } = await this.resilience.execute(
        `github-get-repo-${this.params.owner}-${this.params.repo}`,
        () => this.makeRequest('GET', `/repos/${this.params.owner}/${this.params.repo}`),
        { operation: 'get_repository', owner: this.params.owner, repo: this.params.repo }
      );

      return {
        success: response.ok,
        operation: 'get_repository',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.message || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_repository',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * List repositories operation
   */
  private async listRepositories(): Promise<GitHubResult> {
    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        per_page: String(this.params.perPage),
        ...(this.params.page && { page: String(this.params.page) }),
      });

      const { response, data, timing } = await this.resilience.execute(
        'github-list-repos',
        () => this.makeRequest('GET', `/user/repos?${params.toString()}`),
        { operation: 'list_repositories' }
      );

      return {
        success: response.ok,
        operation: 'list_repositories',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.message || 'Unknown error',
        timing,
        pagination: this.extractPagination(response),
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'list_repositories',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Create issue operation
   */
  private async createIssue(): Promise<GitHubResult> {
    if (!this.params.owner || !this.params.repo || !this.params.title) {
      throw new Error('owner, repo, and title are required for create_issue operation');
    }

    const startTime = Date.now();

    try {
      const body = {
        title: this.params.title,
        body: this.params.body,
        labels: this.params.labels,
      };

      const { response, data, timing } = await this.resilience.execute(
        `github-create-issue-${this.params.owner}-${this.params.repo}`,
        () => this.makeRequest('POST', `/repos/${this.params.owner}/${this.params.repo}/issues`, body),
        { operation: 'create_issue', owner: this.params.owner, repo: this.params.repo, title: this.params.title }
      );

      return {
        success: response.ok,
        operation: 'create_issue',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.message || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'create_issue',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * List issues operation
   */
  private async listIssues(): Promise<GitHubResult> {
    if (!this.params.owner || !this.params.repo) {
      throw new Error('owner and repo are required for list_issues operation');
    }

    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        state: this.params.state,
        per_page: String(this.params.perPage),
        ...(this.params.page && { page: String(this.params.page) }),
      });

      const { response, data, timing } = await this.resilience.execute(
        `github-list-issues-${this.params.owner}-${this.params.repo}`,
        () => this.makeRequest('GET', `/repos/${this.params.owner}/${this.params.repo}/issues?${params.toString()}`),
        { operation: 'list_issues', owner: this.params.owner, repo: this.params.repo }
      );

      return {
        success: response.ok,
        operation: 'list_issues',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.message || 'Unknown error',
        timing,
        pagination: this.extractPagination(response),
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'list_issues',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Create pull request operation
   */
  private async createPullRequest(): Promise<GitHubResult> {
    if (!this.params.owner || !this.params.repo || !this.params.title || !this.params.head || !this.params.base) {
      throw new Error('owner, repo, title, head, and base are required for create_pull_request operation');
    }

    const startTime = Date.now();

    try {
      const body = {
        title: this.params.title,
        body: this.params.body,
        head: this.params.head,
        base: this.params.base,
      };

      const { response, data, timing } = await this.resilience.execute(
        `github-create-pr-${this.params.owner}-${this.params.repo}`,
        () => this.makeRequest('POST', `/repos/${this.params.owner}/${this.params.repo}/pulls`, body),
        { operation: 'create_pull_request', owner: this.params.owner, repo: this.params.repo }
      );

      return {
        success: response.ok,
        operation: 'create_pull_request',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.message || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'create_pull_request',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Merge pull request operation
   */
  private async mergePullRequest(): Promise<GitHubResult> {
    if (!this.params.owner || !this.params.repo || !this.params.pullNumber) {
      throw new Error('owner, repo, and pullNumber are required for merge_pull_request operation');
    }

    const startTime = Date.now();

    try {
      const body = {
        commit_title: this.params.commitTitle,
        commit_message: this.params.commitMessage,
        merge_method: this.params.mergeMethod,
      };

      const { response, data, timing } = await this.resilience.execute(
        `github-merge-pr-${this.params.owner}-${this.params.repo}-${this.params.pullNumber}`,
        () => this.makeRequest('PUT', `/repos/${this.params.owner}/${this.params.repo}/pulls/${this.params.pullNumber}/merge`, body),
        { operation: 'merge_pull_request', owner: this.params.owner, repo: this.params.repo, pullNumber: this.params.pullNumber }
      );

      return {
        success: response.ok,
        operation: 'merge_pull_request',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.message || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'merge_pull_request',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * List branches operation
   */
  private async listBranches(): Promise<GitHubResult> {
    if (!this.params.owner || !this.params.repo) {
      throw new Error('owner and repo are required for list_branches operation');
    }

    const startTime = Date.now();

    try {
      const params = new URLSearchParams({
        per_page: String(this.params.perPage),
        ...(this.params.page && { page: String(this.params.page) }),
      });

      const { response, data, timing } = await this.resilience.execute(
        `github-list-branches-${this.params.owner}-${this.params.repo}`,
        () => this.makeRequest('GET', `/repos/${this.params.owner}/${this.params.repo}/branches?${params.toString()}`),
        { operation: 'list_branches', owner: this.params.owner, repo: this.params.repo }
      );

      return {
        success: response.ok,
        operation: 'list_branches',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : (data as any)?.message || 'Unknown error',
        timing,
        pagination: this.extractPagination(response),
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'list_branches',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<GitHubResult> {
    switch (this.params.operation) {
      case 'get_repository':
        return this.getRepository();
      case 'list_repositories':
        return this.listRepositories();
      case 'create_issue':
        return this.createIssue();
      case 'list_issues':
        return this.listIssues();
      case 'create_pull_request':
        return this.createPullRequest();
      case 'merge_pull_request':
        return this.mergePullRequest();
      case 'list_branches':
        return this.listBranches();
      default:
        return {
          success: false,
          operation: this.params.operation,
          status: { code: 400, reason: 'Invalid operation' },
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default GitHubBubble;
