"use strict";
/**
 * LeanAide Proof Verifier
 *
 * Verifies Lean proofs using LeanAide (AI-assisted theorem proving).
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.LeanAideVerifier = void 0;
const logger_1 = require("../../../lib/logger");
const circuit_breaker_1 = require("../../../lib/circuit-breaker");
/**
 * LeanAide Verifier
 */
class LeanAideVerifier {
    constructor(apiUrl, timeout = 60000) {
        this.systemName = 'leanaide';
        this.apiUrl = apiUrl;
        this.timeout = timeout;
        this.loggerContext = {
            correlation_id: `leanaide-verifier-${Date.now()}`,
            source_service: 'leanaide-verifier',
            target_service: 'leanaide-api',
        };
        // Initialize circuit breaker
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: 5,
            timeout_ms: 60000,
            onStateChange: (old, newState) => {
                logger_1.logger.warn('LeanAide circuit breaker state changed', {
                    ...this.loggerContext,
                    old_state: old,
                    new_state: newState,
                });
            },
        });
        logger_1.logger.info('LeanAide Verifier initialized', {
            ...this.loggerContext,
            api_url: apiUrl,
            timeout,
        });
    }
    /**
     * Verify a proof with LeanAide
     */
    async verify(request) {
        const startTime = Date.now();
        logger_1.logger.info('Verifying proof with LeanAide', {
            ...this.loggerContext,
            request_id: request.requestId,
        });
        try {
            // Execute through circuit breaker
            const result = await this.circuitBreaker.execute(async () => {
                return await this.executeLeanAide(request);
            });
            logger_1.logger.info('LeanAide verification completed', {
                ...this.loggerContext,
                verified: result.verified,
                confidence: result.confidence,
                duration_ms: result.executionTime,
            });
            return result;
        }
        catch (error) {
            logger_1.logger.error('LeanAide verification failed', error, this.loggerContext);
            return {
                system: 'leanaide',
                verified: false,
                confidence: 0.0,
                output: '',
                executionTime: Date.now() - startTime,
                errorMessage: error instanceof Error ? error.message : String(error),
                timestamp: new Date().toISOString(),
            };
        }
    }
    /**
     * Execute LeanAide proof verification
     */
    async executeLeanAide(request) {
        const requestBody = {
            theorem: request.problem.statement,
            library: request.problem.metadata?.library || 'Mathlib',
            timeout: Math.min(request.constraints.timeout, this.timeout) / 1000,
        };
        const response = await fetch(`${this.apiUrl}/api/v1/verify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
            signal: AbortSignal.timeout(this.timeout),
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'LeanAide API request failed');
        }
        const data = await response.json();
        const executionTime = Date.now() - startTime;
        // Determine verification result
        const verified = data.is_valid === true;
        const confidence = data.is_valid ? 0.95 : 0.3;
        const output = data.tactics ? JSON.stringify(data.tactics, null, 2) : '';
        return {
            system: 'leanaide',
            verified,
            confidence,
            output,
            proof: data.proof,
            executionTime,
            timestamp: new Date().toISOString(),
        };
    }
    /**
     * Check if LeanAide service is healthy
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.apiUrl}/health`, {
                signal: AbortSignal.timeout(5000),
            });
            return response.ok;
        }
        catch {
            return false;
        }
    }
}
exports.LeanAideVerifier = LeanAideVerifier;
//# sourceMappingURL=leanaide-verifier.js.map