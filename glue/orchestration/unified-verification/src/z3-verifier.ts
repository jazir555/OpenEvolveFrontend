/**
 * Z3 Proof Verifier
 *
 * Verifies SMT-LIB formatted proofs using Z3 SMT solver.
 */

import { logger, LoggerContext } from '../../../lib/logger';
import { CircuitBreaker } from '../../../lib/circuit-breaker';
import { retryWithBackoff } from '../../../lib/retry';
import type { ProofVerifier, SystemResult } from './verification-service';
import type { VerificationRequest } from './canonical';

/**
 * Z3 API response
 */
interface Z3ApiResponse {
  result?: 'sat' | 'unsat' | 'unknown';
  proof?: string;
  model?: string;
  time?: number;
  error?: string;
}

/**
 * Z3 Verifier
 */
export class Z3Verifier implements ProofVerifier {
  systemName: 'z3' = 'z3';
  circuitBreaker: CircuitBreaker;

  private apiUrl: string;
  private timeout: number;
  private loggerContext: LoggerContext;

  constructor(apiUrl: string, timeout: number = 30000) {
    this.apiUrl = apiUrl;
    this.timeout = timeout;

    this.loggerContext = {
      correlation_id: `z3-verifier-${Date.now()}`,
      source_service: 'z3-verifier',
      target_service: 'z3-api',
    };

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: 60000,
      onStateChange: (old, newState) => {
        logger.warn('Z3 circuit breaker state changed', {
          ...this.loggerContext,
          old_state: old,
          new_state: newState,
        });
      },
    });

    logger.info('Z3 Verifier initialized', {
      ...this.loggerContext,
      api_url: apiUrl,
      timeout,
    });
  }

  /**
   * Verify a proof with Z3
   */
  async verify(request: VerificationRequest): Promise<SystemResult> {
    const startTime = Date.now();

    logger.info('Verifying proof with Z3', {
      ...this.loggerContext,
      request_id: request.requestId,
    });

    try {
      // Execute through circuit breaker
      const result = await this.circuitBreaker.execute(async () => {
        return await this.executeZ3(request);
      });

      logger.info('Z3 verification completed', {
        ...this.loggerContext,
        verified: result.verified,
        confidence: result.confidence,
        duration_ms: result.executionTime,
      });

      return result;
    } catch (error) {
      logger.error('Z3 verification failed', error as Error, this.loggerContext);

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
  private async executeZ3(request: VerificationRequest): Promise<SystemResult> {
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

    const data: Z3ApiResponse = await response.json();
    const executionTime = Date.now() - startTime;

    // Determine verification result
    let verified = false;
    let confidence = 0.0;
    let output = '';

    if (data.result === 'sat') {
      verified = true;
      confidence = 0.95;
      output = data.model || 'Satisfiable';
    } else if (data.result === 'unsat') {
      verified = true;
      confidence = 1.0;
      output = 'Unsatisfiable';
    } else {
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
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.apiUrl}/health`, {
        signal: AbortSignal.timeout(5000),
      });

      return response.ok;
    } catch {
      return false;
    }
  }
}
