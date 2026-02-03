/**
 * Automated Changelog Generator - SECURITY HARDENED (Wave 2)
 */
import { BubbleFlow, HttpBubble, AIAgentBubble, type WebhookEvent } from '@bubblelab/bubble-core';
import { z } from 'zod';
import {
  validateEnvironment, authenticateRequest, requireAuthentication, RateLimiter, InputValidator,
  StructuredLogger, generateCorrelationId, SecuritySchemas,
} from '../security-utils';

const RepositorySchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_\-\/]+$/, 'Invalid repository');
const BranchSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_\-\/]+$/, 'Invalid branch');

validateEnvironment({
  required: ['GITHUB_PAT', 'OPENAI_API_KEY', 'API_KEY'],
  optional: ['SLACK_WEBHOOK_URL'],
  schemas: { API_KEY: SecuritySchemas.apiKey, GITHUB_PAT: SecuritySchemas.token },
});

interface ChangelogEntry {
  version: string;
  date: string;
  changes: { added: string[]; changed: string[]; fixed: string[]; removed: string[]; security: string[] };
  contributors: string[];
}

export class AutomatedChangelogGenerator extends BubbleFlow<'webhook/http'> {
  readonly name = 'Automated Changelog Generator';
  readonly description = 'Generate and maintain changelogs from commit history';
  private rateLimiter = new RateLimiter({ maxRequests: 40, windowMs: 60000 });
  private logger = new StructuredLogger('automated-changelog-generator');

  async handle(payload: WebhookEvent & any): Promise<any> {
    const correlationId = generateCorrelationId();
    if (!this.rateLimiter.checkLimit(correlationId)) throw new Error('Rate limit exceeded');
    const authContext = authenticateRequest(payload.headers?.['x-api-key'], process.env.API_KEY, { correlationId });
    requireAuthentication(authContext);

    const repository = RepositorySchema.parse(payload.repository);
    const branch = BranchSchema.parse(payload.branch || 'main');

    this.logger.info({ msg: 'Generating changelog', correlationId, repository, branch });

    const commits = await this.getCommits(repository, branch, correlationId);
    const tags = await this.getTags(repository, correlationId);

    const currentVersion = tags[0]?.name.replace(/^v/, '') || '1.0.0';
    const previousVersion = tags[1]?.name.replace(/^v/, '') || '0.0.0';

    const changes = await this.analyzeCommitsWithAI(repository, branch, commits, correlationId);
    const contributors = [...new Set(commits.map((c: any) => c.author?.login || c.commit?.author?.name))];

    const changelog = this.generateChangelogMarkdown(currentVersion, changes, contributors);

    this.logger.info({ msg: 'Changelog generated', correlationId, version: currentVersion });

    return {
      timestamp: new Date().toISOString(),
      repository,
      branch,
      fromVersion: previousVersion,
      toVersion: currentVersion,
      changelog,
      correlationId,
    };
  }

  private async getCommits(repository: string, branch: string, correlationId: string): Promise<any[]> {
    const getCommits = new HttpBubble({
      url: `https://api.github.com/repos/${repository}/commits?sha=${branch}&per_page=100`,
      method: 'GET',
      headers: { 'Authorization': `token ${process.env.GITHUB_PAT}`, 'Accept': 'application/vnd.github.v3+json' },
      timeout: 15000,
    });
    const response = await getCommits.action();
    return response.data;
  }

  private async getTags(repository: string, correlationId: string): Promise<any[]> {
    const getTags = new HttpBubble({
      url: `https://api.github.com/repos/${repository}/tags`,
      method: 'GET',
      headers: { 'Authorization': `token ${process.env.GITHUB_PAT}`, 'Accept': 'application/vnd.github.v3+json' },
      timeout: 10000,
    });
    const response = await getTags.action();
    return response.data;
  }

  private async analyzeCommitsWithAI(repository: string, branch: string, commits: any[], correlationId: string): Promise<any> {
    try {
      const agent = new AIAgentBubble({
        model: { model: 'openai/gpt-4' },
        systemPrompt: 'Analyze commit messages and categorize changes. Return JSON with added, changed, fixed, removed, security arrays.',
        message: InputValidator.sanitizeString(`
Analyze these commits and generate a changelog:
Repository: ${repository}
Branch: ${branch}
Commits:
${commits.map((c: any) => `${c.sha.substring(0, 7)}: ${InputValidator.sanitizeString(c.commit?.message?.split('\n')[0] || '', 200)}`).join('\n')}
        `.trim(), 10000),
      });
      const result = await agent.action();

      try {
        return JSON.parse(result.data.response);
      } catch {
        return this.fallbackCategorization(commits);
      }
    } catch (error) {
      this.logger.warn({ msg: 'AI analysis failed, using fallback', correlationId }, error);
      return this.fallbackCategorization(commits);
    }
  }

  private fallbackCategorization(commits: any[]): any {
    const changes = { added: [], changed: [], fixed: [], removed: [], security: [] };

    for (const commit of commits) {
      const msg = (commit.commit?.message || '').toLowerCase();
      const firstLine = commit.commit?.message?.split('\n')[0] || '';

      if (msg.includes('feat') || msg.includes('add')) changes.added.push(firstLine);
      else if (msg.includes('fix') || msg.includes('bug')) changes.fixed.push(firstLine);
      else if (msg.includes('security')) changes.security.push(firstLine);
      else if (msg.includes('remove') || msg.includes('deprecate')) changes.removed.push(firstLine);
      else changes.changed.push(firstLine);
    }

    return changes;
  }

  private generateChangelogMarkdown(version: string, changes: any, contributors: string[]): string {
    const date = new Date().toISOString().split('T')[0];

    return `# Changelog

All notable changes to this project will be documented in this file.

## [${version}] - ${date}

${changes.security?.length > 0 ? `
### Security
${changes.security.map((c: string) => `- ${InputValidator.sanitizeString(c, 500)}`).join('\n')}
` : ''}

${changes.added?.length > 0 ? `
### Added
${changes.added.map((c: string) => `- ${InputValidator.sanitizeString(c, 500)}`).join('\n')}
` : ''}

${changes.changed?.length > 0 ? `
### Changed
${changes.changed.map((c: string) => `- ${InputValidator.sanitizeString(c, 500)}`).join('\n')}
` : ''}

${changes.fixed?.length > 0 ? `
### Fixed
${changes.fixed.map((c: string) => `- ${InputValidator.sanitizeString(c, 500)}`).join('\n')}
` : ''}

${changes.removed?.length > 0 ? `
### Removed
${changes.removed.map((c: string) => `- ${InputValidator.sanitizeString(c, 500)}`).join('\n')}
` : ''}

### Contributors
${contributors.map((c: string) => `- @${InputValidator.sanitizeString(c, 100)}`).join('\n')}

---

[Previous Releases](https://github.com/openevolve/frontend/releases)
`;
  }
}

export default AutomatedChangelogGenerator;
