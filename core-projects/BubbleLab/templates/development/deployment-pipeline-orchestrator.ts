/**
 * Deployment Pipeline Orchestrator - SECURITY HARDENED (Wave 2)
 */
import { BubbleFlow, HttpBubble, AIAgentBubble, SlackBubble, type WebhookEvent } from '@bubblelab/bubble-core';
import { z } from 'zod';
import {
  validateEnvironment, authenticateRequest, requireAuthentication, RateLimiter, InputValidator,
  StructuredLogger, generateCorrelationId, SecuritySchemas, validateServiceName, validateUrl,
} from '../security-utils';

const ApplicationSchema = z.string().min(1).max(100).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid application name');
const ImageSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_\-\/:._]+$/, 'Invalid image name');
const TagSchema = z.string().min(1).max(100).regex(/^[a-zA-Z0-9._\-]+$/, 'Invalid tag');
const EnvironmentSchema = z.enum(['staging', 'production'], { invalid_type_error: 'Invalid environment' });

validateEnvironment({
  required: ['KUBERNETES_API', 'KUBERNETES_TOKEN', 'DOCKER_REGISTRY', 'DOCKER_REGISTRY_TOKEN', 'API_KEY'],
  optional: ['SLACK_WEBHOOK_URL', 'BASE_URL'],
  schemas: { API_KEY: SecuritySchemas.apiKey, KUBERNETES_TOKEN: SecuritySchemas.token },
});

interface StageResult { stage: string; status: 'success' | 'failed' | 'skipped'; duration: number; output?: string; }

export class DeploymentPipelineOrchestrator extends BubbleFlow<'webhook/http'> {
  readonly name = 'Deployment Pipeline Orchestrator';
  readonly description = 'Orchestrate multi-stage deployment pipeline';
  private rateLimiter = new RateLimiter({ maxRequests: 10, windowMs: 60000 });
  private logger = new StructuredLogger('deployment-pipeline-orchestrator');

  async handle(payload: WebhookEvent & any): Promise<any> {
    const correlationId = generateCorrelationId();
    if (!this.rateLimiter.checkLimit(correlationId)) throw new Error('Rate limit exceeded');
    const authContext = authenticateRequest(payload.headers?.['x-api-key'], process.env.API_KEY, { correlationId });
    requireAuthentication(authContext);

    const application = ApplicationSchema.parse(payload.application);
    const image = ImageSchema.parse(payload.image);
    const tag = TagSchema.parse(payload.tag);
    const environment = EnvironmentSchema.parse(payload.environment);
    const runTests = payload.runTests !== false;

    this.logger.info({ msg: 'Starting deployment', correlationId, application, environment });

    const startTime = Date.now();
    const deploymentId = 'deploy-' + Date.now();
    const stages: StageResult[] = [];
    let finalStatus: 'success' | 'failed' | 'rolled_back' = 'success';

    try {
      stages.push(await this.runBuildVerification(application, image, tag, correlationId));
      if (stages[stages.length - 1].status === 'failed') throw new Error('Build verification failed');

      if (environment === 'production') {
        stages.push(await this.deployToEnvironment(application, image, tag, 'staging', correlationId));
        if (stages[stages.length - 1].status === 'failed') throw new Error('Staging deployment failed');

        if (runTests) {
          stages.push(await this.runSmokeTests(application, 'staging', correlationId));
          if (stages[stages.length - 1].status === 'failed') {
            finalStatus = 'rolled_back';
            await this.rollbackDeployment(application, 'staging', correlationId);
            throw new Error('Smoke tests failed');
          }
        }
      }

      stages.push(await this.deployToEnvironment(application, image, tag, environment, correlationId));
      if (stages[stages.length - 1].status === 'failed') {
        finalStatus = 'rolled_back';
        await this.rollbackDeployment(application, environment, correlationId);
        throw new Error('Target deployment failed');
      }

      stages.push(await this.runHealthChecks(application, environment, correlationId));
      if (stages[stages.length - 1].status === 'failed') {
        finalStatus = 'rolled_back';
        await this.rollbackDeployment(application, environment, correlationId);
        throw new Error('Health checks failed');
      }

      finalStatus = 'success';
    } catch (error) {
      this.logger.error({ msg: 'Deployment failed', correlationId }, error);
    }

    const result = {
      deploymentId, timestamp: new Date().toISOString(), application, environment, image, tag,
      stages, finalStatus, totalDuration: Date.now() - startTime, correlationId,
    };

    await this.sendDeploymentNotification(result, correlationId);
    return result;
  }

  private async runBuildVerification(application: string, image: string, tag: string, correlationId: string): Promise<StageResult> {
    const start = Date.now();
    try {
      const verify = new HttpBubble({
        url: `${process.env.DOCKER_REGISTRY}/v2/${image}/manifests/${tag}`,
        method: 'GET',
        headers: { 'Authorization': `Bearer ${process.env.DOCKER_REGISTRY_TOKEN}` },
        timeout: 10000,
      });
      await verify.action();
      return { stage: 'Build Verification', status: 'success', duration: Date.now() - start };
    } catch (error) {
      return { stage: 'Build Verification', status: 'failed', duration: Date.now() - start, output: 'Verification failed' };
    }
  }

  private async deployToEnvironment(application: string, image: string, tag: string, environment: string, correlationId: string): Promise<StageResult> {
    const start = Date.now();
    try {
      const deploy = new HttpBubble({
        url: `${process.env.KUBERNETES_API}/apis/apps/v1/namespaces/${environment}/deployments/${application}`,
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
          'Content-Type': 'application/strategic-merge-patch+json',
        },
        body: {
          spec: {
            template: {
              spec: {
                containers: [{ name: application, image: `${image}:${tag}` }],
              },
            },
          },
        },
        timeout: 30000,
      });
      await deploy.action();
      await new Promise(resolve => setTimeout(resolve, 5000));
      return { stage: `Deploy to ${environment}`, status: 'success', duration: Date.now() - start };
    } catch (error) {
      return { stage: `Deploy to ${environment}`, status: 'failed', duration: Date.now() - start, output: 'Deployment failed' };
    }
  }

  private async runSmokeTests(application: string, environment: string, correlationId: string): Promise<StageResult> {
    const start = Date.now();
    try {
      const tests = new HttpBubble({
        url: validateUrl(`${process.env.BASE_URL}/${environment}/${application}/tests/smoke`),
        method: 'POST',
        timeout: 60000,
      });
      const response = await tests.action();
      return { stage: `Smoke Tests (${environment})`, status: response.data.success ? 'success' : 'failed', duration: Date.now() - start };
    } catch (error) {
      return { stage: `Smoke Tests (${environment})`, status: 'failed', duration: Date.now() - start, output: 'Tests failed' };
    }
  }

  private async runHealthChecks(application: string, environment: string, correlationId: string): Promise<StageResult> {
    const start = Date.now();
    try {
      const healthCheck = new HttpBubble({
        url: validateUrl(`${process.env.BASE_URL}/${environment}/${application}/health`),
        method: 'GET',
        timeout: 10000,
      });
      const response = await healthCheck.action();
      return { stage: `Health Checks (${environment})`, status: response.status === 200 ? 'success' : 'failed', duration: Date.now() - start };
    } catch (error) {
      return { stage: `Health Checks (${environment})`, status: 'failed', duration: Date.now() - start, output: 'Health check failed' };
    }
  }

  private async rollbackDeployment(application: string, environment: string, correlationId: string): Promise<void> {
    try {
      const rollback = new HttpBubble({
        url: `${process.env.KUBERNETES_API}/apis/apps/v1/namespaces/${environment}/deployments/${application}/rollback`,
        method: 'POST',
        headers: { 'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`, 'Content-Type': 'application/json' },
        body: {},
        timeout: 30000,
      });
      await rollback.action();
      this.logger.info({ msg: 'Rollback completed', correlationId, application, environment });
    } catch (error) {
      this.logger.error({ msg: 'Rollback failed', correlationId }, error);
    }
  }

  private async sendDeploymentNotification(result: any, correlationId: string): Promise<void> {
    if (!process.env.SLACK_WEBHOOK_URL) return;
    try {
      const statusEmoji = result.finalStatus === 'success' ? '✅' : result.finalStatus === 'rolled_back' ? '⏪' : '❌';
      const message = InputValidator.sanitizeString(
        `${statusEmoji} Deployment ${result.finalStatus.toUpperCase()}\n\nApplication: ${result.application}\nEnvironment: ${result.environment}\nImage: ${result.image}:${result.tag}\nDuration: ${(result.totalDuration / 1000).toFixed(2)}s`,
        2000
      );
      const slack = new SlackBubble({ webhookUrl: process.env.SLACK_WEBHOOK_URL, message });
      await slack.action();
    } catch (error) {
      this.logger.warn({ msg: 'Slack notification failed', correlationId }, error);
    }
  }
}

export default DeploymentPipelineOrchestrator;
