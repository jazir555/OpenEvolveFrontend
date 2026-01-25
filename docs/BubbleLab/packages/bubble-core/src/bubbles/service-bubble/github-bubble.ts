import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * GitHubBubble - GitHub repository and operations management
 */
export class GitHubBubble extends ServiceBubble<GitHubParams, GitHubResult> {
  bubbleName = 'github';
  type = 'service';
  alias = 'GitHub';
  credentialType = 'github_api_key';

  params = {
    token: z.string().min(1),
    baseUrl: z.string().url().default('https://api.github.com'),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { Octokit } = await import('octokit');
    this.client = new Octokit({
      auth: this.params.token,
      baseUrl: this.params.baseUrl,
      throttle: { enabled: false }
    });
  }

  async getRepository(params: { owner: string; repo: string }): Promise<GitHubResult> {
    try {
      const result = await this.client.rest.repos.get({
        owner: params.owner,
        repo: params.repo
      });
      return { success: true, repository: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createIssue(params: { owner: string; repo: string; title: string; body?: string; labels?: string[] }): Promise<GitHubResult> {
    try {
      const result = await this.client.rest.issues.create({
        owner: params.owner,
        repo: params.repo,
        title: params.title,
        body: params.body,
        labels: params.labels
      });
      return { success: true, issue: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createPullRequest(params: { owner: string; repo: string; title: string; head: string; base: string; body?: string }): Promise<GitHubResult> {
    try {
      const result = await this.client.rest.pulls.create({
        owner: params.owner,
        repo: params.repo,
        title: params.title,
        head: params.head,
        base: params.base,
        body: params.body
      });
      return { success: true, pullRequest: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async mergePullRequest(params: { owner: string; repo: string; pullNumber: number; commitTitle?: string; commitMessage?: string }): Promise<GitHubResult> {
    try {
      const result = await this.client.rest.pulls.merge({
        owner: params.owner,
        repo: params.repo,
        pull_number: params.pullNumber,
        commit_title: params.commitTitle,
        commit_message: params.commitMessage
      });
      return { success: true, mergeResult: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async listIssues(params: { owner: string; repo: string; state?: string; labels?: string }): Promise<GitHubResult> {
    try {
      const result = await this.client.rest.issues.listForRepo({
        owner: params.owner,
        repo: params.repo,
        state: params.state || 'open',
        labels: params.labels
      });
      return { success: true, issues: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getBranches(params: { owner: string; repo: string; protected?: boolean }): Promise<GitHubResult> {
    try {
      const result = await this.client.rest.repos.listBranches({
        owner: params.owner,
        repo: params.repo,
        protected: params.protected
      });
      return { success: true, branches: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createWebhook(params: { owner: string; repo: string; name: string; config: any; events?: string[] }): Promise<GitHubResult> {
    try {
      const result = await this.client.rest.repos.createWebhook({
        owner: params.owner,
        repo: params.repo,
        name: params.name,
        config: params.config,
        events: params.events || ['push']
      });
      return { success: true, webhook: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getFileContents(params: { owner: string; repo: string; path: string; ref?: string }): Promise<GitHubResult> {
    try {
      const result = await this.client.rest.repos.getContent({
        owner: params.owner,
        repo: params.repo,
        path: params.path,
        ref: params.ref
      });
      return { success: true, content: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface GitHubParams {
  token: string;
  baseUrl?: string;
  timeout?: number;
}

export interface GitHubResult {
  success: boolean;
  repository?: any;
  issue?: any;
  pullRequest?: any;
  mergeResult?: any;
  issues?: any[];
  branches?: any[];
  webhook?: any;
  content?: any;
  error?: string;
}
