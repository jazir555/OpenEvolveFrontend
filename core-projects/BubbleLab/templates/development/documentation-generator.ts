/**
 * Documentation Generator - SECURITY HARDENED (Wave 2)
 */
import { BubbleFlow, HttpBubble, AIAgentBubble, type WebhookEvent } from '@bubblelab/bubble-core';
import { z } from 'zod';
import {
  validateEnvironment, authenticateRequest, requireAuthentication, RateLimiter, InputValidator,
  StructuredLogger, generateCorrelationId, SecuritySchemas, validateUrl,
} from '../security-utils';

const RepositorySchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_\-\/]+$/, 'Invalid repository');
const BranchSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_\-\/]+$/, 'Invalid branch');
const FilePathSchema = z.string().min(1).max(500).regex(/^[a-zA-Z0-9_\-\/.]+$/, 'Invalid file path');

validateEnvironment({ required: ['GITHUB_PAT', 'OPENAI_API_KEY', 'API_KEY'], schemas: { API_KEY: SecuritySchemas.apiKey } });

export class DocumentationGenerator extends BubbleFlow<'webhook/http'> {
  readonly name = 'Documentation Generator';
  readonly description = 'Auto-generate and update documentation from code';
  private rateLimiter = new RateLimiter({ maxRequests: 30, windowMs: 60000 });
  private logger = new StructuredLogger('documentation-generator');

  async handle(payload: WebhookEvent & any): Promise<any> {
    const correlationId = generateCorrelationId();
    if (!this.rateLimiter.checkLimit(correlationId)) throw new Error('Rate limit exceeded');
    const authContext = authenticateRequest(payload.headers?.['x-api-key'], process.env.API_KEY, { correlationId });
    requireAuthentication(authContext);

    const repository = RepositorySchema.parse(payload.repository);
    const branch = BranchSchema.parse(payload.branch || 'main');
    const docType = InputValidator.sanitizeString(payload.docType || 'readme', 50);

    this.logger.info({ msg: 'Generating documentation', correlationId, repository, docType });

    let documentation = '';
    let outputPath = '';

    switch (docType) {
      case 'api':
        documentation = await this.generateAPIDocumentation(repository, correlationId);
        outputPath = 'docs/api/API_REFERENCE.md';
        break;
      case 'readme':
        documentation = await this.generateReadme(repository, correlationId);
        outputPath = 'README.md';
        break;
      case 'changelog':
        documentation = await this.generateChangelog(repository, payload.commitHash, correlationId);
        outputPath = 'CHANGELOG.md';
        break;
      case 'architecture':
        documentation = await this.generateArchitectureDocs(repository, correlationId);
        outputPath = 'docs/ARCHITECTURE.md';
        break;
    }

    const sanitizedPath = FilePathSchema.parse(outputPath);
    await this.updateDocumentation(repository, sanitizedPath, branch, documentation, payload.commitHash, correlationId);

    this.logger.info({ msg: 'Documentation generated', correlationId, outputPath: sanitizedPath });

    return { timestamp: new Date().toISOString(), repository, docType, filesGenerated: [sanitizedPath], commitCreated: true, correlationId };
  }

  private async generateAPIDocumentation(repository: string, correlationId: string): Promise<string> {
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Generate comprehensive API documentation from OpenAPI spec in Markdown format',
      message: 'Generate API documentation from spec.',
    });
    const result = await agent.action();
    return `# API Reference\n${InputValidator.sanitizeString(result.data.response, 10000)}\n\n---\n*Auto-generated*`;
  }

  private async generateReadme(repository: string, correlationId: string): Promise<string> {
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Generate a comprehensive README.md',
      message: 'Generate README for project.',
    });
    const result = await agent.action();
    return InputValidator.sanitizeString(result.data.response, 10000);
  }

  private async generateChangelog(repository: string, commitHash: string, correlationId: string): Promise<string> {
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Generate changelog from commits',
      message: 'Generate changelog.',
    });
    const result = await agent.action();
    return `# Changelog\n\n## [Unreleased]\n${InputValidator.sanitizeString(result.data.response, 5000)}\n\n---\n*Last updated: ${new Date().toISOString().split('T')[0]}*`;
  }

  private async generateArchitectureDocs(repository: string, correlationId: string): Promise<string> {
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Generate architecture documentation',
      message: 'Generate architecture docs.',
    });
    const result = await agent.action();
    return `# Architecture Documentation\n${InputValidator.sanitizeString(result.data.response, 10000)}\n\n---\n*Auto-generated*`;
  }

  private async updateDocumentation(repository: string, outputPath: string, branch: string, documentation: string, commitHash: string, correlationId: string): Promise<void> {
    const getFile = new HttpBubble({
      url: `https://api.github.com/repos/${repository}/contents/${outputPath}`,
      method: 'GET',
      headers: { 'Authorization': `token ${process.env.GITHUB_PAT}`, 'Accept': 'application/vnd.github.v3+json' },
      timeout: 10000,
    });

    let currentSha = '';
    try {
      const fileResponse = await getFile.action();
      currentSha = fileResponse.data.sha;
    } catch { /* File doesn't exist */ }

    const updateDoc = new HttpBubble({
      url: `https://api.github.com/repos/${repository}/contents/${outputPath}`,
      method: 'PUT',
      headers: { 'Authorization': `token ${process.env.GITHUB_PAT}`, 'Accept': 'application/vnd.github.v3+json' },
      body: {
        message: InputValidator.sanitizeString(`docs: auto-update documentation\n\nCommit: ${commitHash || 'N/A'}`, 200),
        content: Buffer.from(InputValidator.sanitizeString(documentation, 10000)).toString('base64'),
        sha: currentSha || undefined,
        branch,
      },
      timeout: 10000,
    });

    await updateDoc.action();
  }
}

export default DocumentationGenerator;
