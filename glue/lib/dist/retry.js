"use strict";
/**
 * Exponential Backoff with Jitter
 *
 * Follows the Federation Constitution:
 * - Failure Management: Transient failures get exponential backoff with jitter
 * - Law of Configuration Explicitness: All timeouts configurable
 */
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.retryWithBackoff = retryWithBackoff;
var logger_1 = require("./logger");
var DEFAULT_RETRY_OPTIONS = {
    max_retries: 3,
    base_delay_ms: 1000,
    max_delay_ms: 30000,
    jitter_ms: 500,
    onRetry: undefined,
};
/**
 * Calculate delay with exponential backoff and jitter
 *
 * Formula: min(base_delay * 2^attempt + random_jitter, max_delay)
 */
function calculateDelay(attempt, base_delay_ms, max_delay_ms, jitter_ms) {
    var exponentialDelay = base_delay_ms * Math.pow(2, attempt);
    var jitter = Math.random() * jitter_ms;
    var delay = exponentialDelay + jitter;
    return Math.min(delay, max_delay_ms);
}
/**
 * Retry a function with exponential backoff and jitter
 *
 * Use this for transient failures (network blips, temporary timeouts)
 * Do not use for logic failures (bad data should go to DLQ)
 *
 * @param fn - Async function to retry
 * @param options - Retry configuration
 * @returns Result of successful function execution
 * @throws Last error if all retries exhausted
 */
function retryWithBackoff(fn, options) {
    return __awaiter(this, void 0, void 0, function () {
        var config, lastError, attempt, error_1, delay;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    config = __assign(__assign({}, DEFAULT_RETRY_OPTIONS), options);
                    attempt = 0;
                    _a.label = 1;
                case 1:
                    if (!(attempt <= config.max_retries)) return [3 /*break*/, 7];
                    _a.label = 2;
                case 2:
                    _a.trys.push([2, 4, , 6]);
                    return [4 /*yield*/, fn()];
                case 3: return [2 /*return*/, _a.sent()];
                case 4:
                    error_1 = _a.sent();
                    lastError = error_1 instanceof Error ? error_1 : new Error(String(error_1));
                    // If this was the last attempt, don't delay
                    if (attempt === config.max_retries) {
                        return [3 /*break*/, 7];
                    }
                    delay = calculateDelay(attempt, config.base_delay_ms, config.max_delay_ms, config.jitter_ms);
                    logger_1.logger.warn('Retrying after error', {
                        attempt: attempt + 1,
                        max_retries: config.max_retries + 1,
                        delay_ms: Math.round(delay),
                        error_name: lastError.name,
                        error_message: lastError.message,
                    });
                    // Call onRetry callback if provided
                    if (config.onRetry) {
                        config.onRetry(attempt + 1, lastError);
                    }
                    // Wait before retrying
                    return [4 /*yield*/, sleep(delay)];
                case 5:
                    // Wait before retrying
                    _a.sent();
                    return [3 /*break*/, 6];
                case 6:
                    attempt++;
                    return [3 /*break*/, 1];
                case 7:
                    // All retries exhausted
                    logger_1.logger.error('All retries exhausted', lastError, {
                        max_retries: config.max_retries + 1,
                    });
                    throw lastError;
            }
        });
    });
}
/**
 * Sleep for specified milliseconds
 */
function sleep(ms) {
    return new Promise(function (resolve) { return setTimeout(resolve, ms); });
}
/**
 * Example usage:
 *
 * ```typescript
 * import { retryWithBackoff } from './retry';
 *
 * // Simple retry with defaults
 * const result = await retryWithBackoff(
 *   async () => {
 *     const response = await fetch('http://service:8000/api');
 *     if (!response.ok) throw new Error('HTTP error');
 *     return response.json();
 *   },
 *   { max_retries: 3 }
 * );
 *
 * // Custom retry configuration
 * const data = await retryWithBackoff(
 *   async () => {
 *     return await apiClient.getData();
 *   },
 *   {
 *     max_retries: 5,
 *     base_delay_ms: 500,
 *     max_delay_ms: 10000,
 *     jitter_ms: 200,
 *     onRetry: (attempt, error) => {
 *       console.log(`Attempt ${attempt} failed: ${error.message}`);
 *     },
 *   }
 * );
 *
 * // Retry with correlation ID
 * const result = await retryWithBackoff(
 *   async () => {
 *     logger.info('Calling external API', {
 *       correlation_id: ctx.id,
 *       target_service: 'external-api',
 *     });
 *     return await externalApiCall();
 *   },
 *   { max_retries: 3 }
 * );
 * ```
 */
