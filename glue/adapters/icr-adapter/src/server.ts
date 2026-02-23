/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Adapter Server
 *
 * Express server that provides a REST API for all 7 ICR modes.
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Observability: Structured logging with correlation IDs
 */

import express from 'express';
import { v4 as uuidv4 } from 'uuid';
import { icrAdapter } from './adapter';
import { Logger } from '../../lib/logger';

const app = express();
const logger = new Logger('icr-adapter-server');
const port = process.env.SERVICE_PORT || 3002;

app.use(express.json());

// Logging middleware
app.use((req, res, next) => {
  const correlationId = req.headers['x-correlation-id'] || uuidv4();
  (req as any).correlationId = correlationId;

  logger.info({
    msg: 'Incoming request',
    method: req.method,
    url: req.url,
    correlation_id: correlationId
  });

  next();
});

// Health check
app.get('/health', async (req, res) => {
  try {
    const health = await icrAdapter.healthCheck();
    res.status(200).json(health);
  } catch (error) {
    res.status(500).json({ status: 'unhealthy', error: String(error) });
  }
});

// Mode execution endpoint
app.post('/api/modes/execute', async (req, res) => {
  const { mode, prompt, options } = req.body;
  const { correlationId } = (req as any);

  if (!mode || !prompt) {
    return res.status(400).json({ error: 'Missing mode or prompt' });
  }

  try {
    let result;
    switch (mode) {
      case 'refine':
        result = await icrAdapter.createRefinementRequest(prompt, options, correlationId);
        break;
      case 'react':
        result = await icrAdapter.createReactRequest(prompt, options, correlationId);
        break;
      case 'deepthink':
        result = await icrAdapter.createDeepthinkRequest(prompt, options, correlationId);
        break;
      case 'adaptive_deepthink':
        result = await icrAdapter.createAdaptiveDeepthinkRequest(prompt, options, correlationId);
        break;
      case 'agentic':
        result = await icrAdapter.createAgenticRequest(prompt, options, correlationId);
        break;
      case 'contextual':
        // Use memory-enhanced contextual request by default if available
        if (icrAdapter.hasMemoryAgent()) {
          result = await icrAdapter.createContextualRequestWithMemory(prompt, { ...options, enable_learning: true }, correlationId);
        } else {
          result = await icrAdapter.createContextualRequest(prompt, options, correlationId);
        }
        break;
      case 'generative_ui':
        result = await icrAdapter.createGenerativeUIRequest(prompt, options, correlationId);
        break;
      default:
        return res.status(400).json({ error: `Unknown mode: ${mode}` });
    }

    res.status(200).json(result);
  } catch (error) {
    logger.error({
      msg: 'Mode execution failed',
      mode,
      error: String(error),
      correlation_id: correlationId
    });
    res.status(500).json({ error: String(error) });
  }
});

app.listen(port, () => {
  logger.info({
    msg: 'ICR Adapter Server started',
    port,
    timestamp_utc: new Date().toISOString()
  });
});
