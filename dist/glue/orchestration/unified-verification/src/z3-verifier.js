"use strict";
/**
 * Z3 Proof Verifier
 *
 * Verifies SMT-LIB formatted proofs using Z3 SMT solver.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Z3Verifier = void 0;
const logger_1 = require("../../../lib/logger");
const circuit_breaker_1 = require("../../../lib/circuit-breaker");
/**
 * Z3 Verifier
 */
class Z3Verifier {
    constructor(apiUrl, timeout = 30000) {
        this.systemName = 'z3';
        this.apiUrl = apiUrl;
        this.timeout = timeout;
        this.loggerContext = {
            correlation_id: `z3-verifier-${Date.now()}`,
            source_service: 'z3-verifier',
            target_service: 'z3-api',
        };
        // Initialize circuit breaker
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: 5,
            timeout_ms: 60000,
            onStateChange: (old, newState) => {
                logger_1.logger.warn('Z3 circuit breaker state changed', {
                    ...this.loggerContext,
                    old_state: old,
                    new_state: newState,
                });
            },
        });
        logger_1.logger.info('Z3 Verifier initialized', {
            ...this.loggerContext,
            api_url: apiUrl,
            timeout,
        });
    }
    /**
     * Verify a proof with Z3
     */
    async verify(request) {
        const startTime = Date.now();
        logger_1.logger.info('Verifying proof with Z3', {
            ...this.loggerContext,
            request_id: request.requestId,
        });
        try {
            // Execute through circuit breaker
            const result = await this.circuitBreaker.execute(async () => {
                return await this.executeZ3(request);
            });
            logger_1.logger.info('Z3 verification completed', {
                ...this.loggerContext,
                verified: result.verified,
                confidence: result.confidence,
                duration_ms: result.executionTime,
            });
            return result;
        }
        catch (error) {
            logger_1.logger.error('Z3 verification failed', error, this.loggerContext);
            return {
                system: 'z3',
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
     * Execute Z3 proof
     */
    async executeZ3(request) {
        const requestBody = {
            problem: request.problem.statement,
            timeout: Math.min(request.constraints.timeout, this.timeout) / 1000,
        };
        const response = await fetch(`${this.apiUrl}/api/v1/solve`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
            signal: AbortSignal.timeout(this.timeout),
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Z3 API request failed');
        }
        const data = await response.json();
        const executionTime = Date.now() - startTime;
        // Determine verification result
        let verified = false;
        let confidence = 0.0;
        let output = '';
        if (data.result === 'sat') {
            verified = true;
            confidence = 0.95;
            output = data.model || 'Satisfiable';
        }
        else if (data.result === 'unsat') {
            verified = true;
            confidence = 1.0;
            output = 'Unsatisfiable';
        }
        else {
            verified = false;
            confidence = 0.5;
            output = 'Unknown';
        }
        return {
            system: 'z3',
            verified,
            confidence,
            output: JSON.stringify(data, null, 2),
            proof: data.proof,
            executionTime,
            timestamp: new Date().toISOString(),
        };
    }
    /**
     * Check if Z3 service is healthy
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
exports.Z3Verifier = Z3Verifier;
//# sourceMappingURL=z3-verifier.js.map