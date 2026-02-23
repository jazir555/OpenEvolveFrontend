/**
 * Unified Verification Service
 *
 * Federation Constitution - Unified Verification System (ADR-007)
 *
 * Orchestrates proof verification across multiple systems (Z3, LeanAide, etc.)
 * using canonical proof format.
 *
 * Features:
 * - Single API for all proof systems
 * - Cross-validation between systems
 * - Dependency tracking and revalidation
 * - Circuit breaker protection per system
 * - Retry with exponential backoff
 */

import { v4 as uuidv4 } from 'uuid';
import { logger, LoggerContext } from '../../lib/logger';
import { CircuitBreaker } from '../../lib/circuit-breaker';
import { retryWithBackoff } from '../../lib/retry';
import type {
  VerificationRequest,
  VerificationOptions,
  CrossValidationResult,
  VerificationResult,
  SystemResult,
  ConfidenceScore,
} from './canonical';

/**
 * Result from executing a proof in a native system
 */
interface ExecutionResult {
  success: boolean;
  output?: string;
  proof?: string;
  executionTime?: number;
  error?: string;
}

/**
 * Proof verifier interface
 * All system verifiers must implement this
 */
export interface ProofVerifier {
  systemName: 'z3' | 'leanaide' | 'lean' | 'coq';
  circuitBreaker?: CircuitBreaker;

  // Verify a proof
  verify(request: VerificationRequest): Promise<SystemResult>;

  // Check if system is available
  healthCheck(): Promise<boolean>;
}

/**
 * Unified Verification Service
 */
export class UnifiedVerificationService {
  private verifiers: Map<string, ProofVerifier> = new Map();
  private loggerContext: LoggerContext;

  constructor() {
    this.loggerContext = {
      correlation_id: `verification-${Date.now()}`,
      source_service: 'unified-verification',
    };

    logger.info('Unified Verification Service initialized', this.loggerContext);
  }

  /**
   * Register a proof verifier
   */
  registerVerifier(verifier: ProofVerifier): void {
    this.verifiers.set(verifier.systemName, verifier);

    logger.info('Registered proof verifier', {
      ...this.loggerContext,
      system: verifier.systemName,
    });
  }

  /**
   * Verify a proof using specified strategy
   */
  async verifyProof(
    request: VerificationRequest,
    options: VerificationOptions = {}
  ): Promise<CrossValidationResult> {
    const correlationId = options.correlationId || request.correlationId || uuidv4();
    const ctx: LoggerContext = { ...this.loggerContext, correlation_id };

    logger.info('Starting proof verification', {
      ...ctx,
      request_id: request.requestId,
      problem_type: request.problem.type,
      strategy: options.strategy || 'parallel',
    });

    const startTime = Date.now();

    try {
      // Determine verification strategy
      const strategy = options.strategy || this.determineStrategy(request);

      // Execute verification based on strategy
      const systemResults = await this.executeStrategy(request, strategy, ctx);

      // Cross-validate results
      const crossValidation = this.crossValidate(systemResults, request);

      // Calculate confidence score
      const confidence = this.calculateConfidence(crossValidation, systemResults);

      const result: CrossValidationResult = {
        requestId: request.requestId,
        verified: crossValidation.resolution === 'verified',
        agreement: crossValidation.agreement,
        agreementType: crossValidation.agreementType,
        confidence: confidence.combined,
        systemResults: systemResults.map(r => this.formatResult(r)),
        conflicts: crossValidation.conflicts,
        resolution: crossValidation.resolution as any,
        strategy,
        metadata: {
          correlationId,
          totalExecutionTime: Date.now() - startTime,
          timestamp: new Date().toISOString(),
        },
      };

      logger.info('Proof verification completed', {
        ...ctx,
        verified: result.verified,
        confidence: result.confidence,
        resolution: result.resolution,
        duration_ms: result.metadata.totalExecutionTime,
      });

      return result;
    } catch (error) {
      logger.error('Proof verification failed', error as Error, ctx);

      throw error;
    }
  }

  /**
   * Batch verify multiple proofs
   */
  async batchVerify(
    requests: VerificationRequest[],
    options: VerificationOptions = {}
  ): Promise<CrossValidationResult[]> {
    const correlationId = options.correlationId || uuidv4();
    const ctx: LoggerContext = { ...this.loggerContext, correlation_id };

    logger.info('Starting batch proof verification', {
      ...ctx,
      batch_size: requests.length,
    });

    // Verify in parallel if option allows
    if (options.parallel !== false) {
      const results = await Promise.all(
        requests.map(req => this.verifyProof(req, { ...options, correlationId }))
      );

      return results;
    }
    // Sequential verification
    const results: CrossValidationResult[] = [];
    for (const req of requests) {
      const result = await this.verifyProof(req, { ...options, correlationId });
      results.push(result);
    }

    return results;
  }

  /**
   * Revalidate proofs when a dependency changes
   */
  async revalidateOnDependencyChange(
    changedProofId: string,
    dependentProofs: VerificationRequest[],
    options: VerificationOptions = {}
  ): Promise<CrossValidationResult[]> {
    const correlationId = options.correlationId || uuidv4();
    const ctx: LoggerContext = { ...this.loggerContext, correlation_id };

    logger.info('Revalidating on dependency change', {
      ...ctx,
      changed_proof_id: changedProofId,
      dependent_count: dependentProofs.length,
    });

    // Revalidate all dependent proofs
    const results = await this.batchVerify(dependentProofs, {
      ...options,
      correlationId,
    });

    return results;
  }

  /**
   * Execute verification strategy
   */
  private async executeStrategy(
    request: VerificationRequest,
    strategy: string,
    ctx: LoggerContext
  ): Promise<SystemResult[]> {
    switch (strategy) {
      case 'z3_only':
        return await this.verifyWithSystem(request, 'z3', ctx);

      case 'leanaide_only':
        return await this.verifyWithSystem(request, 'leanaide', ctx);

      case 'parallel':
        return await this.verifyInParallel(request, ctx);

      case 'sequential':
        return await this.verifySequentially(request, ctx);

      case 'hybrid':
        return await this.verifyHybrid(request, ctx);

      default:
        throw new Error(`Unknown strategy: ${strategy}`);
    }
  }

  /**
   * Verify with a single system
   */
  private async verifyWithSystem(
    request: VerificationRequest,
    system: 'z3' | 'leanaide',
    ctx: LoggerContext
  ): Promise<SystemResult[]> {
    const verifier = this.verifiers.get(system);

    if (!verifier) {
      throw new Error(`Verifier not found for system: ${system}`);
    }

    logger.debug('Verifying with system', {
      ...ctx,
      system,
    });

    try {
      const result = await retryWithBackoff(
        () => verifier.verify(request),
        { max_retries: 3, base_delay_ms: 1000 }
      );

      return [result];
    } catch (error) {
      logger.error('System verification failed', error as Error, {
        ...ctx,
        system,
      });

      return [{
        system,
        verified: false,
        confidence: 0.0,
        output: '',
        executionTime: 0,
        errorMessage: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString(),
      }];
    }
  }

  /**
   * Verify in parallel (all systems simultaneously)
   */
  private async verifyInParallel(
    request: VerificationRequest,
    ctx: LoggerContext
  ): Promise<SystemResult[]> {
    logger.debug('Verifying in parallel', ctx);

    const systems = Array.from(this.verifiers.keys());
    const results = await Promise.all(
      systems.map(system =>
        this.verifyWithSystem(request, system as 'z3' | 'leanaide', ctx).then(results => results[0])
      )
    );

    return results;
  }

  /**
   * Verify sequentially (systems one after another)
   */
  private async verifySequentially(
    request: VerificationRequest,
    ctx: LoggerContext
  ): Promise<SystemResult[]> {
    logger.debug('Verifying sequentially', ctx);

    const results: SystemResult[] = [];

    for (const [systemName] of this.verifiers) {
      const systemResults = await this.verifyWithSystem(
        request,
        systemName as 'z3' | 'leanaide',
        ctx
      );
      results.push(...systemResults);

      // Stop if first system proves it
      if (systemResults[0].verified && systemResults[0].confidence > 0.95) {
        logger.debug('Sequential verification: system proved theorem, stopping', {
          ...ctx,
          system: systemName,
          confidence: systemResults[0].confidence,
        });
        break;
      }
    }

    return results;
  }

  /**
   * Hybrid verification: Z3 first, then LeanAide if needed
   */
  private async verifyHybrid(
    request: VerificationRequest,
    ctx: LoggerContext
  ): Promise<SystemResult[]> {
    logger.debug('Verifying with hybrid strategy', ctx);

    const results: SystemResult[] = [];

    // Try Z3 first (faster for SMT problems)
    const z3Results = await this.verifyWithSystem(request, 'z3', ctx);
    results.push(...z3Results);

    // If Z3 fails with high confidence or is inconclusive, try LeanAide
    if (!z3Results[0].verified || z3Results[0].confidence < 0.8) {
      logger.debug('Z3 inconclusive, trying LeanAide', ctx);

      const leanaideResults = await this.verifyWithSystem(request, 'leanaide', ctx);
      results.push(...leanaideResults);
    }

    return results;
  }

  /**
   * Cross-validate results from multiple systems
   */
  private crossValidate(systemResults: SystemResult[], request: VerificationRequest) {
    const verified = systemResults.filter(r => r.verified);
    const unverified = systemResults.filter(r => !r.verified);

    // All systems agree
    if (verified.length === systemResults.length) {
      return {
        agreement: true,
        agreementType: 'full_agreement' as const,
        confidenceAlignment: this.confidencesAlign(systemResults),
        verificationAlignment: true,
        details: 'All systems verified the proof',
        conflicts: [],
        resolution: 'verified' as const,
      };
    }

    // All systems agree it's not verifiable
    if (unverified.length === systemResults.length) {
      return {
        agreement: true,
        agreementType: 'disagreement' as const,
        confidenceAlignment: this.confidencesAlign(systemResults),
        verificationAlignment: true,
        details: 'All systems failed to verify the proof',
        conflicts: [],
        resolution: 'not_verified' as const,
      };
    }

    // Mixed results
    return {
      agreement: false,
      agreementType: 'partial_agreement' as const,
      confidenceAlignment: this.confidencesAlign(systemResults),
      verificationAlignment: false,
      details: 'Systems disagree on verification outcome',
      conflicts: this.detectConflicts(systemResults),
      resolution: this.determineResolution(verified, unverified, request),
    };
  }

  /**
   * Calculate combined confidence score
   */
  private calculateConfidence(
    crossValidation: ReturnType<typeof UnifiedVerificationService.prototype['crossValidate']>,
    systemResults: SystemResult[]
  ): ConfidenceScore {
    const individual: Record<string, number> = {};
    const weights: Record<string, number> = {};

    // Assign weights based on system performance
    let totalWeight = 0;
    for (const result of systemResults) {
      const weight = this.getSystemWeight(result.system);
      weights[result.system] = weight;
      individual[result.system] = result.confidence;
      totalWeight += weight;
    }

    // Calculate weighted average
    let weightedSum = 0;
    for (const result of systemResults) {
      weightedSum += result.confidence * weights[result.system];
    }

    const combined = totalWeight > 0 ? weightedSum / totalWeight : 0;

    return {
      combined,
      individual,
      weights,
      evidence: [
        {
          source: 'cross_validation',
          weight: 1.0,
          description: `Cross-validation from ${systemResults.length} systems`,
        },
      ],
      meetsThreshold: combined >= 0.95,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get system weight for confidence calculation
   */
  private getSystemWeight(system: string): number {
    // Z3 is more reliable for SMT problems
    if (system === 'z3') return 0.6;
    if (system === 'leanaide') return 0.4;
    return 0.5;
  }

  /**
   * Check if confidence scores align across systems
   */
  private confidencesAlign(systemResults: SystemResult[]): boolean {
    if (systemResults.length < 2) return true;

    const confidences = systemResults.map(r => r.confidence);
    const max = Math.max(...confidences);
    const min = Math.min(...confidences);

    // Consider aligned if within 20%
    return (max - min) < 0.2;
  }

  /**
   * Detect conflicts between system results
   */
  private detectConflicts(systemResults: SystemResult[]) {
    const conflicts = [];

    // Check for verification outcome conflicts
    const verified = systemResults.filter(r => r.verified);
    const unverified = systemResults.filter(r => !r.verified);

    if (verified.length > 0 && unverified.length > 0) {
      conflicts.push({
        type: 'verification_outcome' as const,
        systemA: verified[0].system,
        systemB: unverified[0].system,
        description: `${verified[0].system} verified, ${unverified[0].system} did not`,
        severity: 'high' as const,
        resolution: 'trust_higher_confidence' as const,
      });
    }

    // Check for confidence level conflicts
    if (!this.confidencesAlign(systemResults)) {
      const maxConf = Math.max(...systemResults.map(r => r.confidence));
      const minConf = Math.min(...systemResults.map(r => r.confidence));

      if (maxConf - minConf > 0.3) {
        conflicts.push({
          type: 'confidence_level' as const,
          systemA: systemResults.find(r => r.confidence === maxConf)!.system,
          systemB: systemResults.find(r => r.confidence === minConf)!.system,
          description: `Confidence mismatch: ${maxConf.toFixed(2)} vs ${minConf.toFixed(2)}`,
          severity: 'medium' as const,
          resolution: 'trust_higher_confidence' as const,
        });
      }
    }

    return conflicts;
  }

  /**
   * Determine resolution when systems disagree
   */
  private determineResolution(
    verified: SystemResult[],
    unverified: SystemResult[],
    request: VerificationRequest
  ): 'verified' | 'not_verified' | 'inconclusive' | 'requires_review' | 'escalated' {
    // If we have verified results with high confidence, trust them
    const highConfidenceVerified = verified.filter(r => r.confidence >= 0.9);
    if (highConfidenceVerified.length > 0) {
      return 'verified';
    }

    // If all results have low confidence, require review
    const allLowConfidence = systemResults.every(r => r.confidence < 0.7);
    if (allLowConfidence) {
      return 'inconclusive';
    }

    // More verified than unverified, trust the majority
    if (verified.length > unverified.length) {
      return 'verified';
    }

    // More unverified than verified
    if (unverified.length > verified.length) {
      return 'not_verified';
    }

    // Tie - require manual review
    return 'requires_review';
  }

  /**
   * Determine best verification strategy for a given problem
   */
  private determineStrategy(request: VerificationRequest): string {
    const problemType = request.problem.type;

    // SMT problems -> Z3 first
    if (problemType === 'SMT_CONSTRAINTS' || problemType === 'SAT_SOLVING') {
      return request.constraints.allowedSystems.includes('z3') ? 'z3_only' : 'leanaide_only';
    }

    // Theorem proving -> LeanAide first
    if (problemType === 'THEOREM_PROVING') {
      return request.constraints.allowedSystems.includes('leanaide') ? 'leanaide_only' : 'z3_only';
    }

    // Default to parallel for other types
    return 'parallel';
  }

  /**
   * Format system result for API response
   */
  private formatResult(result: SystemResult): VerificationResult {
    return {
      system: result.system,
      verified: result.verified,
      confidence: result.confidence,
      output: result.output || '',
      proof: result.proof,
      metadata: {
        executionTime: result.executionTime || 0,
        memoryUsed: result.memoryUsed,
        strategy: 'parallel',
        timestamp: result.timestamp,
        errorMessage: result.error,
      },
    };
  }
}
