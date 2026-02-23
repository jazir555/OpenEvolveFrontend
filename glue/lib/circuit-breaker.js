"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CircuitBreaker = exports.CircuitState = void 0;
const logger_1 = require("./logger");
var CircuitState;
(function (CircuitState) {
    CircuitState["CLOSED"] = "closed";
    CircuitState["OPEN"] = "open";
    CircuitState["HALF_OPEN"] = "half_open";
})(CircuitState || (exports.CircuitState = CircuitState = {}));
const DEFAULT_OPTIONS = {
    threshold: 5,
    timeout_ms: 60000,
    reset_timeout_ms: 10000,
};
class CircuitBreaker {
    constructor(options) {
        this.state = CircuitState.CLOSED;
        this.failure_count = 0;
        this.success_count = 0;
        this.last_state_change = new Date();
        this.options = {
            ...DEFAULT_OPTIONS,
            ...options,
        };
    }
    async execute(fn) {
        if (this.state === CircuitState.OPEN) {
            if (this.shouldAttemptReset()) {
                this.transitionTo(CircuitState.HALF_OPEN);
            }
            else {
                const waitTime = this.next_attempt_time
                    ? Math.max(0, this.next_attempt_time.getTime() - Date.now())
                    : this.options.timeout_ms;
                throw new Error(`Circuit breaker is OPEN. Rejecting request. Try again in ${Math.round(waitTime)}ms.`);
            }
        }
        try {
            const result = await fn();
            this.onSuccess();
            return result;
        }
        catch (error) {
            this.onFailure();
            throw error;
        }
    }
    onSuccess() {
        this.success_count++;
        if (this.state === CircuitState.HALF_OPEN) {
            this.transitionTo(CircuitState.CLOSED);
            this.failure_count = 0;
        }
    }
    onFailure() {
        this.failure_count++;
        this.last_failure_time = new Date();
        if (this.state === CircuitState.HALF_OPEN) {
            this.transitionTo(CircuitState.OPEN);
        }
        else if (this.failure_count >= this.options.threshold) {
            this.transitionTo(CircuitState.OPEN);
        }
        logger_1.logger.warn('Circuit breaker failure recorded', {
            state: this.state,
            failure_count: this.failure_count,
            threshold: this.options.threshold,
        });
    }
    shouldAttemptReset() {
        if (!this.next_attempt_time) {
            return false;
        }
        return Date.now() >= this.next_attempt_time.getTime();
    }
    transitionTo(newState) {
        const oldState = this.state;
        this.state = newState;
        this.last_state_change = new Date();
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
        if (this.options.onStateChange) {
            this.options.onStateChange(oldState, newState);
        }
    }
    getState() {
        return this.state;
    }
    getStats() {
        return {
            state: this.state,
            failure_count: this.failure_count,
            success_count: this.success_count,
            last_failure_time: this.last_failure_time,
            last_state_change: this.last_state_change,
        };
    }
    reset() {
        this.transitionTo(CircuitState.CLOSED);
        this.failure_count = 0;
        this.success_count = 0;
        this.last_failure_time = undefined;
        this.next_attempt_time = undefined;
    }
}
exports.CircuitBreaker = CircuitBreaker;
//# sourceMappingURL=circuit-breaker.js.map