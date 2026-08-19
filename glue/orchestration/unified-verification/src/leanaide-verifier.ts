/**
 * LeanAide Proof Verifier
 *
 * Verifies Lean proofs using LeanAide (AI-assisted theorem proving).
 */

import { logger, LoggerContext } from './lib/logger';
import { CircuitBreaker } from './lib/circuit-breaker';
import { retryWithBackoff } from './lib/retry';
import type { ProofVerifier, SystemResult } from './verification-service';
import type { VerificationRequest } from './canonical';

/**
 * LeanAide API response
 */
interface LeanAideApiResponse {
  is_valid?: boolean;
  proof?: string;
  tactics?: string[];
  errors?: Array<{
    message: string;
    line?: number;
  }>;
  time?: number;
}

/**
 * LeanAide Verifier
 */
export class LeanAideVerifier implements ProofVerifier {
  systemName: 'leanaide' = 'leanaide';
  circuitBreaker: CircuitBreaker;

  private apiUrl: string;
  private timeout: number;
  private loggerContext: LoggerContext;

  constructor(apiUrl: string, timeout: number = 60000) {
    this.apiUrl = apiUrl;
    this.timeout = timeout;

    this.loggerContext = {
      correlation_id: `leanaide-verifier-${Date.now()}`,
      source_service: 'leanaide-verifier',
      target_service: 'leanaide-api',
    };

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: 60000,
      onStateChange: (old, newState) => {
        logger.warn('LeanAide circuit breaker state changed', {
          ...this.loggerContext,
          old_state: old,
          new_state: newState,
        });
      },
    });

    logger.info('LeanAide Verifier initialized', {
      ...this.loggerContext,
      api_url: apiUrl,
      timeout,
    });
  }

  /**
   * Verify a proof with LeanAide
   */
  async verify(request: VerificationRequest): Promise<SystemResult> {
    const startTime = Date.now();

    logger.info('Verifying proof with LeanAide', {
      ...this.loggerContext,
      request_id: request.requestId,
    });

    try {
      // Execute through circuit breaker
      const result = await this.circuitBreaker.execute(async () => {
        return await this.executeLeanAide(request);
      });

      logger.info('LeanAide verification completed', {
        ...this.loggerContext,
        verified: result.verified,
        confidence: result.confidence,
        duration_ms: result.executionTime,
      });

      return result;
    } catch (error) {
      logger.error('LeanAide verification failed', error as Error, this.loggerContext);

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
  private async executeLeanAide(request: VerificationRequest): Promise<SystemResult> {
    const startTime = Date.now();

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
      const error = await response.json() as { message?: string };
      throw new Error(error.message || 'LeanAide API request failed');
    }

    const data: LeanAideApiResponse = await response.json() as LeanAideApiResponse;
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
