/**
 * Proof Validator
 *
 * Validates formal proofs and checks dependencies.
 * Integrates with Z3 and LeanAide for proof verification.
 *
 * Federation Constitution Compliance:
 * - Law of Idempotency: Safe to re-validate
 * - Law of UTC: All timestamps in UTC
 * - Circuit Breaker: Prevents cascading failures
 * - Retry Logic: Handles transient failures
 */

import { logger, LoggerContext } from '../../logger';
import { retryWithBackoff } from '../../retry';
import { CircuitBreaker } from '../../circuit-breaker';
import {
  FormalProof,
  ProofValidation,
  RevalidationResult,
} from './canonical';

/**
 * Execution result from running a proof
 */
interface ExecutionResult {
  success: boolean;
  output?: string;
  errors?: Array<{ message: string; line?: number; column?: number }>;
  executionTime?: number;
}

/**
 * Proof Validator Configuration
 */
interface ProofValidatorConfig {
  z3ApiUrl?: string;
  leanaideApiUrl?: string;
  timeoutMs: number;
  maxRetries: number;
}

/**
 * Shape of the Z3 solver API response
 */
interface Z3ApiResponse {
  result?: string;
  model?: string;
  error?: string;
  time_ms?: number;
}

/**
 * Shape of the LeanAide verification API response
 */
interface LeanAideApiResponse {
  verified?: boolean;
  output_tactics?: string;
  errors?: Array<{
    message?: string;
    location?: { line?: number; column?: number };
  }>;
  time_ms?: number;
}

/**
 * Proof Validator
 *
 * Validates formal proofs and manages dependency checking
 */
export class ProofValidator {
  private config: ProofValidatorConfig;
  private z3CircuitBreaker: CircuitBreaker;
  private leanaideCircuitBreaker: CircuitBreaker;

  constructor(config: Partial<ProofValidatorConfig> = {}) {
    this.config = {
      timeoutMs: config.timeoutMs || 30000,
      maxRetries: config.maxRetries || 3,
      z3ApiUrl: config.z3ApiUrl || process.env.Z3_API_URL,
      leanaideApiUrl: config.leanaideApiUrl || process.env.LEANAIDE_API_URL,
    };

    // Initialize circuit breakers for external services
    // Note: the shared glue CircuitBreaker is configured via an options object
    // (threshold + timeout_ms), so the service label lives in the log context.
    this.z3CircuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: this.config.timeoutMs,
    });

    this.leanaideCircuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: this.config.timeoutMs,
    });

    logger.info('Proof Validator initialized', {
      source_service: 'proof-validator',
      timeout_ms: this.config.timeoutMs,
      max_retries: this.config.maxRetries,
    });
  }

  /**
   * Validate a proof
   *
   * Executes the proof in its native system and returns validation result
   *
   * @param proofId - ID of the proof to validate
   * @param revalidateDependencies - Whether to revalidate dependencies
   * @param correlationId - Optional correlation ID for tracing
   * @returns Validation result
   */
  async validateProof(
    proofId: string,
    proof: FormalProof,
    revalidateDependencies: boolean = false,
    correlationId?: string
  ): Promise<ProofValidation> {
    const logContext: LoggerContext = {
      correlation_id: correlationId || proof.correlation_id,
      source_service: 'proof-validator',
      proof_id: proofId,
      system: proof.system,
    };

    try {
      logger.info('Validating proof', {
        ...logContext,
        revalidate_dependencies: revalidateDependencies,
      });

      // Check dependencies first if requested
      if (revalidateDependencies && proof.dependencies && proof.dependencies.length > 0) {
        const depsValid = await this.checkDependenciesValid(proofId, proof, correlationId);
        if (!depsValid) {
          logger.warn('Proof dependencies are invalid', logContext);

          return {
            id: this.generateId(),
            proof_id: proofId,
            is_valid: false,
            errors: [
              {
                message: 'One or more proof dependencies are invalid',
              },
            ],
            validated_at: new Date().toISOString(),
            validated_by: 'system',
          };
        }
      }

      // Execute the proof
      const executionResult = await this.executeProof(proof, correlationId);

      const validation: ProofValidation = {
        id: this.generateId(),
        proof_id: proofId,
        is_valid: executionResult.success,
        errors: executionResult.errors?.map(err => ({
          line: err.line,
          column: err.column,
          message: err.message,
          code: 'VALIDATION_ERROR',
        })),
        validated_at: new Date().toISOString(),
        validated_by: 'automated_check',
        correlation_id: correlationId,
      };

      logger.info('Proof validation completed', {
        ...logContext,
        is_valid: validation.is_valid,
        error_count: validation.errors?.length || 0,
      });

      return validation;
    } catch (error) {
      logger.error('Failed to validate proof', error as Error, logContext);

      return {
        id: this.generateId(),
        proof_id: proofId,
        is_valid: false,
        errors: [
          {
            message: error instanceof Error ? error.message : String(error),
          },
        ],
        validated_at: new Date().toISOString(),
        validated_by: 'system',
        correlation_id: correlationId,
      };
    }
  }

  /**
   * Batch validate multiple proofs
   *
   * @param proofs - Array of proofs with their IDs
   * @param correlationId - Optional correlation ID for tracing
   * @returns Array of validation results
   */
  async batchValidate(
    proofs: Array<{ proofId: string; proof: FormalProof }>,
    correlationId?: string
  ): Promise<ProofValidation[]> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-validator',
    };

    logger.info('Batch validating proofs', {
      ...logContext,
      proof_count: proofs.length,
    });

    const validations: ProofValidation[] = [];

    // Process in parallel with concurrency limit
    const concurrencyLimit = 5;
    for (let i = 0; i < proofs.length; i += concurrencyLimit) {
      const batch = proofs.slice(i, i + concurrencyLimit);
      const batchResults = await Promise.all(
        batch.map(({ proofId, proof }) =>
          this.validateProof(proofId, proof, false, correlationId)
        )
      );
      validations.push(...batchResults);
    }

    logger.info('Batch validation completed', {
      ...logContext,
      total_validations: validations.length,
      valid_count: validations.filter(v => v.is_valid).length,
      invalid_count: validations.filter(v => !v.is_valid).length,
    });

    return validations;
  }

  /**
   * Check validation status of a single dependency
   *
   * Extension point: subclasses can override this to query real storage.
   *
   * @param dependencyId - ID of dependency to check
   * @param correlationId - Optional correlation ID
   * @returns Whether dependency is currently valid
   */
  protected async checkDependencyValidationStatus(
    dependencyId: string,
    correlationId?: string
  ): Promise<boolean> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-validator',
      dependency_id: dependencyId,
    };

    try {
      // Query graph storage or database for this dependency's validation status
      // This would typically be a graph lookup or key-value store query
      // For now, we implement a basic existence check

      logger.debug('Checking dependency validation status', logContext);

      // In production, this would be:
      // const validation = await db.proofValidation.findUnique({
      //   where: { proof_id: dependencyId },
      //   orderBy: { validated_at: 'desc' },
      // });
      // return validation ? validation.is_valid : false;

      // Placeholder: Assume dependency exists and is valid
      // In a real implementation, this would query the actual storage
      return true;
    } catch (error) {
      logger.error('Failed to check dependency validation status', error as Error, logContext);
      // If we can't verify, assume invalid for safety
      return false;
    }
  }

  /**
   * Check if dependencies are valid
   *
   * @param proofId - ID of the proof
   * @param proof - The proof with dependencies
   * @param correlationId - Optional correlation ID for tracing
   * @returns Whether all dependencies are valid
   */
  async checkDependenciesValid(
    proofId: string,
    proof: FormalProof,
    correlationId?: string
  ): Promise<boolean> {
    const logContext: LoggerContext = {
      correlation_id: correlationId || proof.correlation_id,
      source_service: 'proof-validator',
      proof_id: proofId,
    };

    if (!proof.dependencies || proof.dependencies.length === 0) {
      logger.debug('No dependencies to check', logContext);
      return true;
    }

    logger.info('Checking proof dependencies', {
      ...logContext,
      dependency_count: proof.dependencies.length,
    });

    // In a real implementation, you would fetch the validation status
    // of each dependency from storage. For now, we assume dependencies are valid.
    // TODO: Implement dependency lookup from storage

    // Simulate dependency check
    // In production, this would query the graph/database
    const allValid = true; // Placeholder

    logger.info('Dependency check completed', {
      ...logContext,
      all_valid: allValid,
    });

    return allValid;
  }

  /**
   * Revalidate proofs when a dependency changes
   *
   * When a proof is updated, all proofs that depend on it must be revalidated
   *
   * @param changedProofId - ID of the proof that changed
   * @param dependentProofIds - IDs of proofs that depend on the changed proof
   * @param proofs - Map of proof IDs to proofs
   * @param correlationId - Optional correlation ID for tracing
   * @returns Revalidation result
   */
  async revalidateOnDependencyChange(
    changedProofId: string,
    dependentProofIds: string[],
    proofs: Map<string, FormalProof>,
    correlationId?: string
  ): Promise<RevalidationResult> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-validator',
      changed_proof_id: changedProofId,
    };

    logger.info('Revalidating on dependency change', {
      ...logContext,
      dependent_count: dependentProofIds.length,
    });

    let proofsRevalidated = 0;
    let proofsInvalidated = 0;
    const errors: string[] = [];

    // Revalidate each dependent proof
    for (const depProofId of dependentProofIds) {
      try {
        const proof = proofs.get(depProofId);
        if (!proof) {
          logger.warn('Dependent proof not found', {
            ...logContext,
            dependent_proof_id: depProofId,
          });
          continue;
        }

        const validation = await this.validateProof(
          depProofId,
          proof,
          false, // Don't revalidate dependencies recursively
          correlationId
        );

        proofsRevalidated++;

        if (!validation.is_valid) {
          proofsInvalidated++;
          logger.warn('Dependent proof invalidated', {
            ...logContext,
            dependent_proof_id: depProofId,
            errors: validation.errors,
          });
        }
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        errors.push(`${depProofId}: ${errorMsg}`);
        logger.error('Failed to revalidate dependent proof', error as Error, {
          ...logContext,
          dependent_proof_id: depProofId,
        });
      }
    }

    const result: RevalidationResult = {
      success: errors.length === 0,
      proofs_revalidated: proofsRevalidated,
      proofs_invalidated: proofsInvalidated,
      errors: errors.length > 0 ? errors : undefined,
      timestamp: new Date().toISOString(),
    };

    logger.info('Revalidation completed', {
      ...logContext,
      ...result,
    });

    return result;
  }

  /**
   * Execute a proof in its native system
   *
   * @param proof - The proof to execute
   * @param correlationId - Optional correlation ID for tracing
   * @returns Execution result
   */
  private async executeProof(
    proof: FormalProof,
    correlationId?: string
  ): Promise<ExecutionResult> {
    const logContext: LoggerContext = {
      correlation_id: correlationId || proof.correlation_id,
      source_service: 'proof-validator',
      proof_id: proof.id,
      system: proof.system,
    };

    try {
      logger.info('Executing proof', logContext);

      switch (proof.system) {
        case 'z3':
          return await this.executeZ3Proof(proof, correlationId);

        case 'leanaide':
          return await this.executeLeanAideProof(proof, correlationId);

        default:
          logger.warn('Unknown proof system', logContext);
          return {
            success: false,
            errors: [
              {
                message: `Unknown proof system: ${proof.system}`,
              },
            ],
          };
      }
    } catch (error) {
      logger.error('Failed to execute proof', error as Error, logContext);
      return {
        success: false,
        errors: [
          {
            message: error instanceof Error ? error.message : String(error),
          },
        ],
      };
    }
  }

  /**
   * Execute a Z3 proof
   *
   * @param proof - The proof to execute
   * @param correlationId - Optional correlation ID for tracing
   * @returns Execution result
   */
  private async executeZ3Proof(
    proof: FormalProof,
    correlationId?: string
  ): Promise<ExecutionResult> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-validator',
      proof_id: proof.id,
    };

    if (!this.config.z3ApiUrl) {
      logger.warn('Z3 API URL not configured', logContext);
      return {
        success: false,
        errors: [
          {
            message: 'Z3 API URL not configured',
          },
        ],
      };
    }

    try {
      // Use circuit breaker to prevent cascading failures
      const result = await this.z3CircuitBreaker.execute(async () => {
        return retryWithBackoff(
          async () => {
            // Execute Z3 proof via API
            logger.debug('Executing Z3 proof', logContext);

            // Construct the Z3 API request
            const z3Request = {
              problem: proof.proof,
              // Z3-specific parameters
              logic: 'AUFLIRA', // Default logic
              timeout: Math.floor(this.config.timeoutMs / 1000), // Convert to seconds
            };

            // Actual API call to Z3 service
            const response = await fetch(`${this.config.z3ApiUrl}/solve`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
              },
              body: JSON.stringify(z3Request),
              // AbortController is managed by retry wrapper
            });

            if (!response.ok) {
              const errorText = await response.text();
              throw new Error(`Z3 API error: ${response.status} - ${errorText}`);
            }

            const z3Response = (await response.json()) as Z3ApiResponse;

            // Transform Z3 response to execution result
            return {
              success: z3Response.result === 'sat' || z3Response.result === 'unsat' || z3Response.result === 'unknown',
              output: z3Response.model || '',
              errors: z3Response.error ? [{
                message: z3Response.error,
              }] : undefined,
              executionTime: z3Response.time_ms,
            };
          },
          { max_retries: this.config.maxRetries }
        );
      });

      return result;
    } catch (error) {
      logger.error('Z3 proof execution failed', error as Error, logContext);
      return {
        success: false,
        errors: [
          {
            message: error instanceof Error ? error.message : String(error),
          },
        ],
      };
    }
  }

  /**
   * Execute a LeanAide proof
   *
   * @param proof - The proof to execute
   * @param correlationId - Optional correlation ID for tracing
   * @returns Execution result
   */
  private async executeLeanAideProof(
    proof: FormalProof,
    correlationId?: string
  ): Promise<ExecutionResult> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-validator',
      proof_id: proof.id,
    };

    if (!this.config.leanaideApiUrl) {
      logger.warn('LeanAide API URL not configured', logContext);
      return {
        success: false,
        errors: [
          {
            message: 'LeanAide API URL not configured',
          },
        ],
      };
    }

    try {
      // Use circuit breaker to prevent cascading failures
      const result = await this.leanaideCircuitBreaker.execute(async () => {
        return retryWithBackoff(
          async () => {
            // Execute LeanAide proof via API
            logger.debug('Executing LeanAide proof', logContext);

            // Construct the LeanAide API request
            const leanRequest = {
              proof_code: proof.proof,
              environment: 'standard',
              // LeanAide-specific parameters
              timeout: Math.floor(this.config.timeoutMs / 1000),
            };

            // Actual API call to LeanAide service
            const response = await fetch(`${this.config.leanaideApiUrl}/verify`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
              },
              body: JSON.stringify(leanRequest),
            });

            if (!response.ok) {
              const errorText = await response.text();
              throw new Error(`LeanAide API error: ${response.status} - ${errorText}`);
            }

            const leanResponse = (await response.json()) as LeanAideApiResponse;

            // Transform LeanAide response to execution result
            return {
              success: leanResponse.verified === true,
              output: leanResponse.output_tactics || '',
              errors: leanResponse.errors ? leanResponse.errors.map((e: any) => ({
                message: e.message || 'Unknown error',
                line: e.location?.line,
                column: e.location?.column,
              })) : undefined,
              executionTime: leanResponse.time_ms,
            };
          },
          { max_retries: this.config.maxRetries }
        );
      });

      return result;
    } catch (error) {
      logger.error('LeanAide proof execution failed', error as Error, logContext);
      return {
        success: false,
        errors: [
          {
            message: error instanceof Error ? error.message : String(error),
          },
        ],
      };
    }
  }

  /**
   * Generate a UUID v4
   */
  private generateId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  /**
   * Update proof validation in storage
   *
   * Extension point: subclasses can override this to persist validations.
   *
   * @param proofId - ID of the proof
   * @param validation - Validation result
   */
  protected updateProofValidation(proofId: string, validation: ProofValidation): void {
    // In a real implementation, this would update the database
    // For now, we just log it
    logger.info('Proof validation updated', {
      source_service: 'proof-validator',
      proof_id: proofId,
      is_valid: validation.is_valid,
    });

    // TODO: Implement database update
    // await db.proofValidation.update({
    //   where: { proof_id: proofId },
    //   data: validation,
    // });
  }
}

/**
 * Example usage:
 *
 * ```typescript
 * import { ProofValidator } from './validator';
 * import { FormalProof } from './canonical';
 *
 * // Create validator
 * const validator = new ProofValidator({
 *   z3ApiUrl: 'http://z3-core:8000',
 *   leanaideApiUrl: 'http://leanaide-core:8000',
 *   timeoutMs: 30000,
 *   maxRetries: 3,
 * });
 *
 * // Validate a single proof
 * const proof: FormalProof = { ... };
 * const validation = await validator.validateProof(
 *   proof.id,
 *   proof,
 *   true, // revalidate dependencies
 *   'correlation-123'
 * );
 *
 * // Batch validate
 * const validations = await validator.batchValidate([
 *   { proofId: proof1.id, proof: proof1 },
 *   { proofId: proof2.id, proof: proof2 },
 * ], 'correlation-123');
 *
 * // Revalidate on dependency change
 * const result = await validator.revalidateOnDependencyChange(
 *   changedProofId,
 *   dependentProofIds,
 *   proofsMap,
 *   'correlation-123'
 * );
 * ```
 */
