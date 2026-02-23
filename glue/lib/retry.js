"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.retryWithBackoff = retryWithBackoff;
const logger_1 = require("./logger");
const DEFAULT_RETRY_OPTIONS = {
    max_retries: 3,
    base_delay_ms: 1000,
    max_delay_ms: 30000,
    jitter_ms: 500,
    onRetry: undefined,
};
function calculateDelay(attempt, base_delay_ms, max_delay_ms, jitter_ms) {
    const exponentialDelay = base_delay_ms * Math.pow(2, attempt);
    const jitter = Math.random() * jitter_ms;
    const delay = exponentialDelay + jitter;
    return Math.min(delay, max_delay_ms);
}
async function retryWithBackoff(fn, options) {
    const config = {
        ...DEFAULT_RETRY_OPTIONS,
        ...options,
    };
    let lastError;
    for (let attempt = 0; attempt <= config.max_retries; attempt++) {
        try {
            return await fn();
        }
        catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));
            if (attempt === config.max_retries) {
                break;
            }
            const delay = calculateDelay(attempt, config.base_delay_ms, config.max_delay_ms, config.jitter_ms);
            logger_1.logger.warn('Retrying after error', {
                attempt: attempt + 1,
                max_retries: config.max_retries + 1,
                delay_ms: Math.round(delay),
                error_name: lastError.name,
                error_message: lastError.message,
            });
            if (config.onRetry) {
                config.onRetry(attempt + 1, lastError);
            }
            await sleep(delay);
        }
    }
    logger_1.logger.error('All retries exhausted', lastError, {
        max_retries: config.max_retries + 1,
    });
    throw lastError;
}
function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
//# sourceMappingURL=retry.js.map