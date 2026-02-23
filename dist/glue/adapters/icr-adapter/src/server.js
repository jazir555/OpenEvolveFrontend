"use strict";
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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const uuid_1 = require("uuid");
const adapter_1 = require("./adapter");
const logger_1 = require("../../lib/logger");
const app = (0, express_1.default)();
const logger = new logger_1.Logger('icr-adapter-server');
const port = process.env.SERVICE_PORT || 3002;
app.use(express_1.default.json());
// Logging middleware
app.use((req, res, next) => {
    const correlationId = req.headers['x-correlation-id'] || (0, uuid_1.v4)();
    req.correlationId = correlationId;
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
        const health = await adapter_1.icrAdapter.healthCheck();
        res.status(200).json(health);
    }
    catch (error) {
        res.status(500).json({ status: 'unhealthy', error: String(error) });
    }
});
// Mode execution endpoint
app.post('/api/modes/execute', async (req, res) => {
    const { mode, prompt, options } = req.body;
    const correlationId = req.correlationId;
    if (!mode || !prompt) {
        return res.status(400).json({ error: 'Missing mode or prompt' });
    }
    try {
        let result;
        switch (mode) {
            case 'refine':
                result = await adapter_1.icrAdapter.createRefinementRequest(prompt, options, correlationId);
                break;
            case 'react':
                result = await adapter_1.icrAdapter.createReactRequest(prompt, options, correlationId);
                break;
            case 'deepthink':
                result = await adapter_1.icrAdapter.createDeepthinkRequest(prompt, options, correlationId);
                break;
            case 'adaptive_deepthink':
                result = await adapter_1.icrAdapter.createAdaptiveDeepthinkRequest(prompt, options, correlationId);
                break;
            case 'agentic':
                result = await adapter_1.icrAdapter.createAgenticRequest(prompt, options, correlationId);
                break;
            case 'contextual':
                // Use memory-enhanced contextual request by default if available
                if (adapter_1.icrAdapter.hasMemoryAgent()) {
                    result = await adapter_1.icrAdapter.createContextualRequestWithMemory(prompt, { ...options, enable_learning: true }, correlationId);
                }
                else {
                    result = await adapter_1.icrAdapter.createContextualRequest(prompt, options, correlationId);
                }
                break;
            case 'generative_ui':
                result = await adapter_1.icrAdapter.createGenerativeUIRequest(prompt, options, correlationId);
                break;
            default:
                return res.status(400).json({ error: `Unknown mode: ${mode}` });
        }
        res.status(200).json(result);
    }
    catch (error) {
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
//# sourceMappingURL=server.js.map