import { OpenAPIHono } from '@hono/zod-openapi';
import { env } from '../config/env.js';
import {
  setupErrorHandler,
  validationErrorHook,
} from '../utils/error-handler.js';
import { judgeBatchRoute, judgeRoute } from '../schemas/evolution-judge.js';
import { VisualLLMJudge } from '../services/evolution/visual-judge/visual-llm-judge.js';
import { CredentialType } from '../schemas/index.js';
import { getUserId } from '../middleware/auth.js';
import { trackServiceUsage } from '../services/service-usage-tracking.js';

const app = new OpenAPIHono({
  defaultHook: validationErrorHook,
});
setupErrorHandler(app);

const buildCredentials = () => {
  const credentials = {
    [CredentialType.OPENAI_CRED]: env.OPENAI_API_KEY,
    [CredentialType.ANTHROPIC_CRED]: env.ANTHROPIC_API_KEY,
    [CredentialType.GOOGLE_GEMINI_CRED]: env.GOOGLE_API_KEY,
    [CredentialType.OPENROUTER_CRED]: env.OPENROUTER_API_KEY,
  } as Record<CredentialType, string | undefined>;

  const required = [
    CredentialType.OPENAI_CRED,
    CredentialType.ANTHROPIC_CRED,
    CredentialType.GOOGLE_GEMINI_CRED,
  ];

  const missing = required.filter((key) => !credentials[key]);

  if (missing.length) {
    throw new Error(`Missing API keys: ${missing.join(', ')}`);
  }

  return credentials as Record<CredentialType, string>;
};

app.get('/health', (c) => c.json({ status: 'ok' }));

app.openapi(judgeRoute, async (c) => {
  const userId = getUserId(c);
  const { input, weights } = c.req.valid('json');
  const judge = new VisualLLMJudge();
  const credentials = buildCredentials();

  const result = await judge.evaluate(input, credentials, weights);

  const totalCost = result.agents.reduce(
    (sum, agent) => sum + (agent.costUsd || 0),
    0
  );

  if (totalCost > 0) {
    await trackServiceUsage(userId, {
      service: CredentialType.OPENAI_CRED,
      subService: 'visual-judge',
      unit: 'evaluation',
      usage: 1,
      unitCost: totalCost,
      totalCost,
    });
  }

  return c.json(result, 200);
});

app.openapi(judgeBatchRoute, async (c) => {
  const userId = getUserId(c);
  const { inputs, maxConcurrency } = c.req.valid('json');
  const judge = new VisualLLMJudge();
  const credentials = buildCredentials();

  const results = await judge.evaluateBatch(
    inputs,
    credentials,
    maxConcurrency ?? 3
  );

  const totalCost = results.flatMap((r) => r.agents).reduce(
    (sum, agent) => sum + (agent.costUsd || 0),
    0
  );

  if (totalCost > 0) {
    await trackServiceUsage(userId, {
      service: CredentialType.OPENAI_CRED,
      subService: 'visual-judge',
      unit: 'evaluation',
      usage: results.length,
      unitCost: totalCost / results.length,
      totalCost,
    });
  }

  return c.json(results, 200);
});

export default app;
