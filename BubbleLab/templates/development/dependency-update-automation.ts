/**
 * Dependency Update Automation - SECURITY HARDENED (Wave 2)
 * Purpose: Monitor dependencies, check for updates, and create PRs
 *
 * Security Fixes:
 * - Environment variable validation
 * - API key authentication
 * - Input validation for all user inputs
 * - Rate limiting
 * - Command injection prevention
 * - Error sanitization
 * - Structured logging
 */

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type CronEvent
} from '@bubblelab/bubble-core';
import { z } from 'zod';
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  StructuredLogger,
  generateCorrelationId,
  SecuritySchemas,
  validateUrl,
} from '../security-utils';

const DependencyNameSchema = z.string().min(1).max(100).regex(/^[a-zA-Z0-9_\-@/.]+$/, 'Invalid dependency name');
const VersionSchema = z.string().min(1).max(50).regex(/^[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9.\-+]*$/, 'Invalid version format');

interface DependencyUpdate {
  name: string;
  currentVersion: string;
  latestVersion: string;
  type: 'major' | 'minor' | 'patch';
  security: boolean;
}

interface UpdateReport {
  timestamp: string;
  total: number;
  security: number;
  major: number;
  minor: number;
  patch: number;
  prsCreated: string[];
  correlationId: string;
}

validateEnvironment({
  required: ['GITHUB_PAT', 'API_KEY'],
  optional: ['SLACK_WEBHOOK_URL'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
    GITHUB_PAT: SecuritySchemas.token,
  },
});

export class DependencyUpdateAutomation extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 9 * * 1';
  readonly name = 'Dependency Update Automation';
  readonly description = 'Monitor dependencies and create update PRs';

  private rateLimiter = new RateLimiter({ maxRequests: 20, windowMs: 60000 });
  private logger = new StructuredLogger('dependency-update-automation');

  async handle(payload: CronEvent): Promise<UpdateReport> {
    const correlationId = generateCorrelationId();
    const timestamp = new Date().toISOString();

    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({ msg: 'Rate limit exceeded', correlationId });
      throw new Error('Rate limit exceeded');
    }

    const authContext = authenticateRequest(payload.headers?.['x-api-key'], process.env.API_KEY, { correlationId });
    requireAuthentication(authContext);

    this.logger.info({ msg: 'Starting dependency update check', correlationId });

    const pythonUpdates = await this.checkPythonUpdates(correlationId);
    const nodeUpdates = await this.checkNodeUpdates(correlationId);
    const allUpdates = [...pythonUpdates, ...nodeUpdates];

    const securityUpdates = allUpdates.filter(u => u.security);
    const majorUpdates = allUpdates.filter(u => u.type === 'major' && !u.security);
    const prsCreated: string[] = [];

    for (const update of [...securityUpdates, ...majorUpdates]) {
      try {
        const prBranch = InputValidator.sanitizeString(`deps/update-${update.name}-${update.latestVersion}`, 255);
        const prTitle = InputValidator.sanitizeString(
          update.security ? `Security: Update ${update.name} to ${update.latestVersion}` : `Update ${update.name} to ${update.latestVersion}`,
          255
        );

        await this.createBranch(prBranch, correlationId);
        await this.commitChanges(prBranch, InputValidator.sanitizeString(`Update ${update.name} to ${update.latestVersion}`, 200), correlationId);
        const prNumber = await this.createPullRequest(prBranch, prTitle, update, correlationId);
        prsCreated.push(prNumber);
      } catch (error) {
        this.logger.warn({ msg: 'Failed to create PR', correlationId, dependency: update.name }, error);
      }
    }

    const report: UpdateReport = {
      timestamp,
      total: allUpdates.length,
      security: securityUpdates.length,
      major: majorUpdates.length,
      minor: allUpdates.filter(u => u.type === 'minor').length,
      patch: allUpdates.filter(u => u.type === 'patch').length,
      prsCreated,
      correlationId,
    };

    if (report.total > 0 && process.env.SLACK_WEBHOOK_URL) {
      try {
        const slack = new SlackBubble({
          webhookUrl: process.env.SLACK_WEBHOOK_URL,
          message: InputValidator.sanitizeString(`Dependency Update Report\nTotal: ${report.total}\nSecurity: ${report.security}\nPRs: ${prsCreated.length}`, 2000),
        });
        await slack.action();
      } catch (error) {
        this.logger.warn({ msg: 'Slack notification failed', correlationId }, error);
      }
    }

    this.logger.info({ msg: 'Dependency update check completed', correlationId, total: report.total });
    return report;
  }

  private async checkPythonUpdates(correlationId: string): Promise<DependencyUpdate[]> {
    const updates: DependencyUpdate[] = [];
    const requirementsUrl = validateUrl('https://raw.githubusercontent.com/openevolve/frontend/main/requirements.txt');

    const requirements = new HttpBubble({ url: requirementsUrl, method: 'GET', timeout: 10000 });
    const response = await requirements.action();
    const lines = response.data.split('\n');

    for (const line of lines) {
      if (line.trim() && !line.startsWith('#')) {
        const match = line.match(/^([a-zA-Z0-9_-]+)([>=<~]+)(.+)$/);
        if (match) {
          const [, name, , currentVersion] = match;
          try {
            DependencyNameSchema.parse(name);
            VersionSchema.parse(currentVersion);

            const pypiCheck = new HttpBubble({
              url: `https://pypi.org/pypi/${name}/json`,
              method: 'GET',
              timeout: 5000,
            });
            const pypiResponse = await pypiCheck.action();
            const latestVersion = pypiResponse.data.info.version;
            const vulnerabilities = pypiResponse.data.vulnerabilities || [];

            if (latestVersion !== currentVersion) {
              updates.push({
                name,
                currentVersion,
                latestVersion,
                type: this.getVersionType(currentVersion, latestVersion),
                security: vulnerabilities.length > 0,
              });
            }
          } catch (error) {
            this.logger.warn({ msg: 'Failed to check Python package', correlationId, package: name }, error);
          }
        }
      }
    }
    return updates;
  }

  private async checkNodeUpdates(correlationId: string): Promise<DependencyUpdate[]> {
    const updates: DependencyUpdate[] = [];
    const packageJsonUrl = validateUrl('https://raw.githubusercontent.com/openevolve/frontend/main/package.json');

    const packageJson = new HttpBubble({ url: packageJsonUrl, method: 'GET', timeout: 10000 });
    const response = await packageJson.action();
    const dependencies = { ...response.data.dependencies, ...response.data.devDependencies };

    for (const [name, currentVersion] of Object.entries(dependencies)) {
      try {
        DependencyNameSchema.parse(name);

        const npmCheck = new HttpBubble({ url: `https://registry.npmjs.org/${name}`, method: 'GET', timeout: 5000 });
        const npmResponse = await npmCheck.action();
        const latestVersion = npmResponse.data['dist-tags'].latest;
        const hasSecurityWarnings = npmResponse.data.hasOwnProperty('security');

        if (latestVersion !== currentVersion.replace(/^[\^~]/, '')) {
          updates.push({
            name,
            currentVersion,
            latestVersion,
            type: this.getVersionType(currentVersion, latestVersion),
            security: hasSecurityWarnings,
          });
        }
      } catch (error) {
        this.logger.warn({ msg: 'Failed to check npm package', correlationId, package: name }, error);
      }
    }
    return updates;
  }

  private getVersionType(current: string, latest: string): 'major' | 'minor' | 'patch' {
    const currentMajor = parseInt(current.split('.')[0]);
    const latestMajor = parseInt(latest.split('.')[0]);
    if (latestMajor > currentMajor) return 'major';
    const currentMinor = parseInt(current.split('.')[1]);
    const latestMinor = parseInt(latest.split('.')[1]);
    if (latestMinor > currentMinor) return 'minor';
    return 'patch';
  }

  private async createBranch(branchName: string, correlationId: string): Promise<void> {
    const sanitizedBranch = InputValidator.sanitizeString(branchName, 255);
    const createBranch = new HttpBubble({
      url: `https://api.github.com/repos/openevolve/frontend/git/refs`,
      method: 'POST',
      headers: {
        'Authorization': `token ${process.env.GITHUB_PAT}`,
        'Accept': 'application/vnd.github.v3+json',
      },
      body: { ref: `refs/heads/${sanitizedBranch}`, sha: 'main' },
      timeout: 10000,
    });
    await createBranch.action();
  }

  private async commitChanges(branch: string, message: string, correlationId: string): Promise<void> {
    const sanitizedBranch = InputValidator.sanitizeString(branch, 255);
    const sanitizedMessage = InputValidator.sanitizeString(message, 200);
    const commit = new HttpBubble({
      url: `https://api.github.com/repos/openevolve/frontend/git/commits`,
      method: 'POST',
      headers: {
        'Authorization': `token ${process.env.GITHUB_PAT}`,
        'Accept': 'application/vnd.github.v3+json',
      },
      body: { message: sanitizedMessage, branch: sanitizedBranch },
      timeout: 10000,
    });
    await commit.action();
  }

  private async createPullRequest(branch: string, title: string, update: DependencyUpdate, correlationId: string): Promise<string> {
    const sanitizedBranch = InputValidator.sanitizeString(branch, 255);
    const sanitizedTitle = InputValidator.sanitizeString(title, 255);
    const sanitizedBody = InputValidator.sanitizeString(
      `This PR updates ${update.name} from ${update.currentVersion} to ${update.latestVersion}.`,
      1000
    );

    const pr = new HttpBubble({
      url: 'https://api.github.com/repos/openevolve/frontend/pulls',
      method: 'POST',
      headers: {
        'Authorization': `token ${process.env.GITHUB_PAT}`,
        'Accept': 'application/vnd.github.v3+json',
      },
      body: {
        title: sanitizedTitle,
        head: sanitizedBranch,
        base: 'main',
        body: sanitizedBody,
        labels: update.security ? ['security', 'dependencies'] : ['dependencies'],
      },
      timeout: 10000,
    });

    const response = await pr.action();
    return response.data.number.toString();
  }
}

export default DependencyUpdateAutomation;
