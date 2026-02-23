"use strict";
/**
 * Circuit Breaker Pattern
 *
 * Follows the Federation Constitution:
 * - Failure Management: System failures trigger circuit breaker
 * - Prevents cascading failures by stopping calls to dead services
 *
 * States:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Circuit is tripped, requests fail immediately
 * - HALF_OPEN: Testing if service has recovered
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
exports.CircuitBreaker = exports.CircuitState = void 0;
var logger_1 = require("./logger");
var CircuitState;
(function (CircuitState) {
    CircuitState["CLOSED"] = "closed";
    CircuitState["OPEN"] = "open";
    CircuitState["HALF_OPEN"] = "half_open";
})(CircuitState || (exports.CircuitState = CircuitState = {}));
var DEFAULT_OPTIONS = {
    threshold: 5,
    timeout_ms: 60000, // 1 minute
    reset_timeout_ms: 10000, // 10 seconds
};
/**
 * Circuit Breaker implementation
 *
 * Prevents cascading failures by stopping calls to failing services
 */
var CircuitBreaker = /** @class */ (function () {
    function CircuitBreaker(options) {
        this.state = CircuitState.CLOSED;
        this.failure_count = 0;
        this.success_count = 0;
        this.last_state_change = new Date();
        this.options = __assign(__assign({}, DEFAULT_OPTIONS), options);
    }
    /**
     * Execute function through circuit breaker
     *
     * @param fn - Async function to execute
     * @returns Result of function execution
     * @throws Error if circuit is OPEN or function fails
     */
    CircuitBreaker.prototype.execute = function (fn) {
        return __awaiter(this, void 0, void 0, function () {
            var waitTime, result, error_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        // Check if we should transition from OPEN to HALF_OPEN
                        if (this.state === CircuitState.OPEN) {
                            if (this.shouldAttemptReset()) {
                                this.transitionTo(CircuitState.HALF_OPEN);
                            }
                            else {
                                waitTime = this.next_attempt_time
                                    ? Math.max(0, this.next_attempt_time.getTime() - Date.now())
                                    : this.options.timeout_ms;
                                throw new Error("Circuit breaker is OPEN. Rejecting request. Try again in ".concat(Math.round(waitTime), "ms."));
                            }
                        }
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 3, , 4]);
                        return [4 /*yield*/, fn()];
                    case 2:
                        result = _a.sent();
                        this.onSuccess();
                        return [2 /*return*/, result];
                    case 3:
                        error_1 = _a.sent();
                        this.onFailure();
                        throw error_1;
                    case 4: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Handle successful execution
     */
    CircuitBreaker.prototype.onSuccess = function () {
        this.success_count++;
        if (this.state === CircuitState.HALF_OPEN) {
            // Service recovered, close the circuit
            this.transitionTo(CircuitState.CLOSED);
            this.failure_count = 0;
        }
    };
    /**
     * Handle failed execution
     */
    CircuitBreaker.prototype.onFailure = function () {
        this.failure_count++;
        this.last_failure_time = new Date();
        if (this.state === CircuitState.HALF_OPEN) {
            // Service still failing, reopen circuit
            this.transitionTo(CircuitState.OPEN);
        }
        else if (this.failure_count >= this.options.threshold) {
            // Threshold reached, trip the circuit
            this.transitionTo(CircuitState.OPEN);
        }
        logger_1.logger.warn('Circuit breaker failure recorded', {
            state: this.state,
            failure_count: this.failure_count,
            threshold: this.options.threshold,
        });
    };
    /**
     * Check if enough time has passed to attempt a reset
     */
    CircuitBreaker.prototype.shouldAttemptReset = function () {
        if (!this.next_attempt_time) {
            return false;
        }
        return Date.now() >= this.next_attempt_time.getTime();
    };
    /**
     * Transition to new state
     */
    CircuitBreaker.prototype.transitionTo = function (newState) {
        var oldState = this.state;
        this.state = newState;
        this.last_state_change = new Date();
        // Set next attempt time when opening circuit
        if (newState === CircuitState.OPEN) {
            this.next_attempt_time = new Date(Date.now() + this.options.timeout_ms);
        }
        else if (newState === CircuitState.CLOSED) {
            this.next_attempt_time = undefined;
        }
        logger_1.logger.info('Circuit breaker state changed', {
            old_state: oldState,
            new_state: newState,
            failure_count: this.failure_count,
        });
        // Call state change callback if provided
        if (this.options.onStateChange) {
            this.options.onStateChange(oldState, newState);
        }
    };
    /**
     * Get current circuit state
     */
    CircuitBreaker.prototype.getState = function () {
        return this.state;
    };
    /**
     * Get circuit breaker statistics
     */
    CircuitBreaker.prototype.getStats = function () {
        return {
            state: this.state,
            failure_count: this.failure_count,
            success_count: this.success_count,
            last_failure_time: this.last_failure_time,
            last_state_change: this.last_state_change,
        };
    };
    /**
     * Manually reset circuit breaker to CLOSED state
     */
    CircuitBreaker.prototype.reset = function () {
        this.transitionTo(CircuitState.CLOSED);
        this.failure_count = 0;
        this.success_count = 0;
        this.last_failure_time = undefined;
        this.next_attempt_time = undefined;
    };
    return CircuitBreaker;
}());
exports.CircuitBreaker = CircuitBreaker;
/**
 * Example usage:
 *
 * ```typescript
 * import { CircuitBreaker } from './circuit-breaker';
 *
 * // Create circuit breaker
 * const cb = new CircuitBreaker({
 *   threshold: 5,           // Trip after 5 failures
 *   timeout_ms: 60000,      // Stay open for 1 minute
 *   onStateChange: (old, newState) => {
 *     console.log(`Circuit: ${old} -> ${newState}`);
 *   },
 * });
 *
 * // Use circuit breaker
 * try {
 *   const result = await cb.execute(async () => {
 *     const response = await fetch('http://service:8000/api');
 *     if (!response.ok) throw new Error('HTTP error');
 *     return response.json();
 *   });
 * } catch (error) {
 *   if (cb.getState() === CircuitState.OPEN) {
 *     logger.error('Service is down, circuit is open', error);
 *     // Use fallback or cached data
 *   } else {
 *     throw error;
 *   }
 * }
 *
 * // Check state
 * const stats = cb.getStats();
 * console.log(stats);
 * // { state: 'closed', failure_count: 2, success_count: 10, ... }
 *
 * // Manual reset (e.g., after health check passes)
 * cb.reset();
 * ```
 */
