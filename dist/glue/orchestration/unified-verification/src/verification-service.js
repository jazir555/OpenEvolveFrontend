"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnifiedVerificationService = void 0;
const uuid_1 = require("uuid");
const logger_1 = require("../../lib/logger");
const retry_1 = require("../../lib/retry");
/**
 * Unified Verification Service
 */
class UnifiedVerificationService {
    constructor() {
        this.verifiers = new Map();
        this.loggerContext = {
            correlation_id: `verification-${Date.now()}`,
            source_service: 'unified-verification',
        };
        logger_1.logger.info('Unified Verification Service initialized', this.loggerContext);
    }
    /**
     * Register a proof verifier
     */
    registerVerifier(verifier) {
        this.verifiers.set(verifier.systemName, verifier);
        logger_1.logger.info('Registered proof verifier', {
            ...this.loggerContext,
            system: verifier.systemName,
        });
    }
    /**
     * Verify a proof using specified strategy
     */
    async verifyProof(request, options = {}) {
        const correlationId = options.correlationId || request.correlationId || (0, uuid_1.v4)();
        const ctx = { ...this.loggerContext, correlation_id };
        logger_1.logger.info('Starting proof verification', {
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
            const result = {
                requestId: request.requestId,
                verified: crossValidation.resolution === 'verified',
                agreement: crossValidation.agreement,
                agreementType: crossValidation.agreementType,
                confidence: confidence.combined,
                systemResults: systemResults.map(r => this.formatResult(r)),
                conflicts: crossValidation.conflicts,
                resolution: crossValidation.resolution,
                strategy,
                metadata: {
                    correlationId,
                    totalExecutionTime: Date.now() - startTime,
                    timestamp: new Date().toISOString(),
                },
            };
            logger_1.logger.info('Proof verification completed', {
                ...ctx,
                verified: result.verified,
                confidence: result.confidence,
                resolution: result.resolution,
                duration_ms: result.metadata.totalExecutionTime,
            });
            return result;
        }
        catch (error) {
            logger_1.logger.error('Proof verification failed', error, ctx);
            throw error;
        }
    }
    /**
     * Batch verify multiple proofs
     */
    async batchVerify(requests, options = {}) {
        const correlationId = options.correlationId || (0, uuid_1.v4)();
        const ctx = { ...this.loggerContext, correlation_id };
        logger_1.logger.info('Starting batch proof verification', {
            ...ctx,
            batch_size: requests.length,
        });
        // Verify in parallel if option allows
        if (options.parallel !== false) {
            const results = await Promise.all(requests.map(req => this.verifyProof(req, { ...options, correlationId })));
            return results;
        }
        else {
            // Sequential verification
            const results = [];
            for (const req of requests) {
                const result = await this.verifyProof(req, { ...options, correlationId });
                results.push(result);
            }
            return results;
        }
    }
    /**
     * Revalidate proofs when a dependency changes
     */
    async revalidateOnDependencyChange(changedProofId, dependentProofs, options = {}) {
        const correlationId = options.correlationId || (0, uuid_1.v4)();
        const ctx = { ...this.loggerContext, correlation_id };
        logger_1.logger.info('Revalidating on dependency change', {
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
    async executeStrategy(request, strategy, ctx) {
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
    async verifyWithSystem(request, system, ctx) {
        const verifier = this.verifiers.get(system);
        if (!verifier) {
            throw new Error(`Verifier not found for system: ${system}`);
        }
        logger_1.logger.debug('Verifying with system', {
            ...ctx,
            system,
        });
        try {
            const result = await (0, retry_1.retryWithBackoff)(() => verifier.verify(request), { max_retries: 3, base_delay_ms: 1000 });
            return [result];
        }
        catch (error) {
            logger_1.logger.error('System verification failed', error, {
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
    async verifyInParallel(request, ctx) {
        logger_1.logger.debug('Verifying in parallel', ctx);
        const systems = Array.from(this.verifiers.keys());
        const results = await Promise.all(systems.map(system => this.verifyWithSystem(request, system, ctx).then(results => results[0])));
        return results;
    }
    /**
     * Verify sequentially (systems one after another)
     */
    async verifySequentially(request, ctx) {
        logger_1.logger.debug('Verifying sequentially', ctx);
        const results = [];
        for (const [systemName] of this.verifiers) {
            const systemResults = await this.verifyWithSystem(request, systemName, ctx);
            results.push(...systemResults);
            // Stop if first system proves it
            if (systemResults[0].verified && systemResults[0].confidence > 0.95) {
                logger_1.logger.debug('Sequential verification: system proved theorem, stopping', {
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
    async verifyHybrid(request, ctx) {
        logger_1.logger.debug('Verifying with hybrid strategy', ctx);
        const results = [];
        // Try Z3 first (faster for SMT problems)
        const z3Results = await this.verifyWithSystem(request, 'z3', ctx);
        results.push(...z3Results);
        // If Z3 fails with high confidence or is inconclusive, try LeanAide
        if (!z3Results[0].verified || z3Results[0].confidence < 0.8) {
            logger_1.logger.debug('Z3 inconclusive, trying LeanAide', ctx);
            const leanaideResults = await this.verifyWithSystem(request, 'leanaide', ctx);
            results.push(...leanaideResults);
        }
        return results;
    }
    /**
     * Cross-validate results from multiple systems
     */
    crossValidate(systemResults, request) {
        const verified = systemResults.filter(r => r.verified);
        const unverified = systemResults.filter(r => !r.verified);
        // All systems agree
        if (verified.length === systemResults.length) {
            return {
                agreement: true,
                agreementType: 'full_agreement',
                confidenceAlignment: this.confidencesAlign(systemResults),
                verificationAlignment: true,
                details: 'All systems verified the proof',
                conflicts: [],
                resolution: 'verified',
            };
        }
        // All systems agree it's not verifiable
        if (unverified.length === systemResults.length) {
            return {
                agreement: true,
                agreementType: 'disagreement',
                confidenceAlignment: this.confidencesAlign(systemResults),
                verificationAlignment: true,
                details: 'All systems failed to verify the proof',
                conflicts: [],
                resolution: 'not_verified',
            };
        }
        // Mixed results
        return {
            agreement: false,
            agreementType: 'partial_agreement',
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
    calculateConfidence(crossValidation, systemResults) {
        const individual = {};
        const weights = {};
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
    getSystemWeight(system) {
        // Z3 is more reliable for SMT problems
        if (system === 'z3')
            return 0.6;
        if (system === 'leanaide')
            return 0.4;
        return 0.5;
    }
    /**
     * Check if confidence scores align across systems
     */
    confidencesAlign(systemResults) {
        if (systemResults.length < 2)
            return true;
        const confidences = systemResults.map(r => r.confidence);
        const max = Math.max(...confidences);
        const min = Math.min(...confidences);
        // Consider aligned if within 20%
        return (max - min) < 0.2;
    }
    /**
     * Detect conflicts between system results
     */
    detectConflicts(systemResults) {
        const conflicts = [];
        // Check for verification outcome conflicts
        const verified = systemResults.filter(r => r.verified);
        const unverified = systemResults.filter(r => !r.verified);
        if (verified.length > 0 && unverified.length > 0) {
            conflicts.push({
                type: 'verification_outcome',
                systemA: verified[0].system,
                systemB: unverified[0].system,
                description: `${verified[0].system} verified, ${unverified[0].system} did not`,
                severity: 'high',
                resolution: 'trust_higher_confidence',
            });
        }
        // Check for confidence level conflicts
        if (!this.confidencesAlign(systemResults)) {
            const maxConf = Math.max(...systemResults.map(r => r.confidence));
            const minConf = Math.min(...systemResults.map(r => r.confidence));
            if (maxConf - minConf > 0.3) {
                conflicts.push({
                    type: 'confidence_level',
                    systemA: systemResults.find(r => r.confidence === maxConf).system,
                    systemB: systemResults.find(r => r.confidence === minConf).system,
                    description: `Confidence mismatch: ${maxConf.toFixed(2)} vs ${minConf.toFixed(2)}`,
                    severity: 'medium',
                    resolution: 'trust_higher_confidence',
                });
            }
        }
        return conflicts;
    }
    /**
     * Determine resolution when systems disagree
     */
    determineResolution(verified, unverified, request) {
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
    determineStrategy(request) {
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
    formatResult(result) {
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
exports.UnifiedVerificationService = UnifiedVerificationService;
//# sourceMappingURL=verification-service.js.map