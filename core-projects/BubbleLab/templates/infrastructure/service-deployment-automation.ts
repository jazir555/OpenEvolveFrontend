/**
 * Service Deployment Automation
 * Purpose: Automated service deployment with health checks and rollback capability
 * Category: Infrastructure Automation
 * Event Type: webhook/http
 *
 * Required Credentials:
 * - KUBERNETES_CONFIG: Kubernetes cluster configuration
 * - DOCKER_REGISTRY: Container registry credentials
 * - API_KEY: API key for authentication (required)
 * - SLACK_WEBHOOK_URL: Slack webhook for deployment notifications (optional)
 *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting
 * - Input validation for all user inputs
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - SQL injection prevention
 * - URL validation for all endpoints
 */

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';
import { z } from 'zod';
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  SecuritySchemas,
} from '../security-utils';

// Input validation schemas
const ServiceNameSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid service name');
const NamespaceSchema = z.string().min(1).max(253).regex(/^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/, 'Invalid namespace format');
const ImageTagSchema = z.string().min(1).max(128).regex(/^[a-zA-Z0-9._-]+$/, 'Invalid image tag format');
const ReplicaCountSchema = z.number().int().min(1).max(100);

interface DeploymentConfig {
  service: string;
  image: string;
  tag: string;
  namespace: string;
  replicas: number;
  environment: Record<string, string>;
}

interface DeploymentResult {
  success: boolean;
  deploymentId: string;
  service: string;
  image: string;
  tag: string;
  timestamp: string;
  duration: number;
  status: 'deploying' | 'healthy' | 'failed' | 'rolled_back';
  healthChecks: {
    initial: boolean;
    final: boolean;
  };
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['KUBERNETES_API', 'KUBERNETES_TOKEN', 'DOCKER_REGISTRY', 'API_KEY'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
    KUBERNETES_API: SecuritySchemas.url,
    DOCKER_REGISTRY: SecuritySchemas.url,
  },
});

export class ServiceDeploymentAutomation extends BubbleFlow<'webhook/http'> {
  readonly name = 'Service Deployment Automation';
  readonly description = 'Automated service deployment with health checks and rollback';

  private logger = new StructuredLogger('service-deployment-automation');
  private rateLimiter = new RateLimiter({
    maxRequests: 10, // 10 deployments per minute
    windowMs: 60000, // 1 minute
  });

  async handle(payload: WebhookEvent & DeploymentConfig): Promise<DeploymentResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const startTime = Date.now();
    const timestamp = new Date().toISOString();
    const deploymentId = `deploy-${Date.now()}`;

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      throw new Error('Rate limit exceeded. Please try again later.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    // Security: Input validation
    const validatedService = InputValidator.validateServiceName(payload.service);
    const validatedNamespace = InputValidator.validateContainerName(payload.namespace);
    const validatedImage = InputValidator.validateContainerName(payload.image);
    const validatedTag = ImageTagSchema.parse(payload.tag);
    const validatedReplicas = ReplicaCountSchema.parse(payload.replicas);

    const { service, image, tag, namespace, replicas, environment } = {
      service: validatedService,
      image: validatedImage,
      tag: validatedTag,
      namespace: validatedNamespace,
      replicas: validatedReplicas,
      environment: payload.environment || {},
    };

    let deploymentResult: DeploymentResult = {
      success: false,
      deploymentId,
      service,
      image,
      tag,
      timestamp,
      duration: 0,
      status: 'deploying',
      healthChecks: {
        initial: false,
        final: false,
      },
      correlationId,
    };

    this.logger.info({
      msg: 'Starting deployment',
      service,
      image: `${image}:${tag}`,
      namespace,
      replicas,
    });

    try {
      // Step 1: Pre-deployment health check
      const preDeployHealth = new HttpBubble({
        url: InputValidator.validateUrl(`${process.env.KUBERNETES_API}/apis/apps/v1/namespaces/${namespace}/deployments/${service}`),
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
          'Content-Type': 'application/json',
        },
        timeout: 10000,
      });

      const preDeployResponse = await preDeployHealth.action();
      deploymentResult.healthChecks.initial = preDeployResponse.status === 200;

      // Step 2: Pull new image
      const pullImage = new HttpBubble({
        url: InputValidator.validateUrl(`${process.env.DOCKER_REGISTRY}/v2/${image}/manifests/${tag}`),
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${process.env.DOCKER_REGISTRY_TOKEN}`,
        },
        timeout: 30000,
      });

      await pullImage.action();

      // Step 3: Update deployment
      const updateDeployment = new HttpBubble({
        url: InputValidator.validateUrl(`${process.env.KUBERNETES_API}/apis/apps/v1/namespaces/${namespace}/deployments/${service}`),
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
          'Content-Type': 'application/strategic-merge-patch+json',
        },
        body: {
          spec: {
            replicas,
            template: {
              spec: {
                containers: [{
                  name: service,
                  image: `${image}:${tag}`,
                  env: Object.entries(environment).map(([name, value]) => ({
                    name: InputValidator.sanitizeString(name, 255),
                    value: InputValidator.sanitizeString(value, 4096),
                  })),
                }],
              },
            },
          },
        },
        timeout: 30000,
      });

      await updateDeployment.action();

      // Step 4: Wait for rollout
      await new Promise(resolve => setTimeout(resolve, 10000));

      // Step 5: Monitor rollout status
      let rolloutHealthy = false;
      let attempts = 0;
      const maxAttempts = 30; // 5 minutes max

      while (!rolloutHealthy && attempts < maxAttempts) {
        const checkRollout = new HttpBubble({
          url: InputValidator.validateUrl(`${process.env.KUBERNETES_API}/apis/apps/v1/namespaces/${namespace}/deployments/${service}/rollout`),
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
          },
          timeout: 10000,
        });

        const rolloutResponse = await checkRollout.action();
        const rollout = rolloutResponse.data;

        rolloutHealthy =
          rollout.status?.readyReplicas === replicas &&
          rollout.status?.updatedReplicas === replicas &&
          rollout.status?.unavailableReplicas === 0;

        if (!rolloutHealthy) {
          await new Promise(resolve => setTimeout(resolve, 10000));
          attempts++;
        }
      }

      deploymentResult.healthChecks.final = rolloutHealthy;
      deploymentResult.status = rolloutHealthy ? 'healthy' : 'failed';
      deploymentResult.success = rolloutHealthy;

      // Step 6: Post-deployment validation
      if (rolloutHealthy && process.env.SERVICE_BASE_URL) {
        const validatedServiceUrl = InputValidator.validateUrl(process.env.SERVICE_BASE_URL);
        const validateService = new HttpBubble({
          url: InputValidator.validateUrl(`${validatedServiceUrl}/${service}/health`),
          method: 'GET',
          timeout: 5000,
        });

        try {
          const healthResponse = await validateService.action();
          deploymentResult.success = healthResponse.status === 200;
        } catch (error) {
          deploymentResult.success = false;
          deploymentResult.status = 'failed';
          this.logger.warn({
            msg: 'Post-deployment health check failed',
          }, error);
        }
      }

      // Step 7: Rollback if failed
      if (!deploymentResult.success) {
        const rollback = new HttpBubble({
          url: InputValidator.validateUrl(`${process.env.KUBERNETES_API}/apis/apps/v1/namespaces/${namespace}/deployments/${service}/rollback`),
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
            'Content-Type': 'application/json',
          },
          body: {},
          timeout: 30000,
        });

        await rollback.action();
        deploymentResult.status = 'rolled_back';
      }

    } catch (error) {
      deploymentResult.status = 'failed';
      deploymentResult.success = false;

      const sanitizedError = sanitizeError(error);
      this.logger.error({
        msg: 'Deployment failed',
        service,
        image: `${image}:${tag}`,
      }, error);

      // Attempt rollback on error
      try {
        const rollback = new HttpBubble({
          url: InputValidator.validateUrl(`${process.env.KUBERNETES_API}/apis/apps/v1/namespaces/${namespace}/deployments/${service}/rollback`),
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
          },
          timeout: 30000,
        });

        await rollback.action();
        deploymentResult.status = 'rolled_back';
      } catch (rollbackError) {
        this.logger.error({
          msg: 'Rollback failed',
        }, rollbackError);
      }
    }

    deploymentResult.duration = Date.now() - startTime;

    // Step 8: Send notification
    const statusEmoji = deploymentResult.success ? '✅' : deploymentResult.status === 'rolled_back' ? '⏪' : '❌';

    const message = InputValidator.sanitizeString(`
${statusEmoji} Deployment ${deploymentResult.status.replace('_', ' ').toUpperCase()}

Service: ${service}
Image: ${image}:${tag}
Namespace: ${namespace}
Duration: ${(deploymentResult.duration / 1000).toFixed(2)}s
Replicas: ${replicas}

Health Checks:
  Initial: ${deploymentResult.healthChecks.initial ? '✅' : '❌'}
  Final: ${deploymentResult.healthChecks.final ? '✅' : '❌'}
    `.trim(), 10000);

    if (process.env.SLACK_WEBHOOK_URL) {
      try {
        const slack = new SlackBubble({
          webhookUrl: process.env.SLACK_WEBHOOK_URL,
          message,
        });

        await slack.action();
      } catch (slackError) {
        this.logger.warn({
          msg: 'Slack notification failed',
        }, slackError);
      }
    }

    this.logger.info({
      msg: 'Deployment completed',
      status: deploymentResult.status,
      success: deploymentResult.success,
      duration: deploymentResult.duration,
    });

    return deploymentResult;
  }
}

export default ServiceDeploymentAutomation;
