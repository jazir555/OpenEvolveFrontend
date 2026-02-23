"use strict";
/**
 * UNIFIED VERIFICATION ORCHESTRATOR
 *
 * Main entry point for unified formal verification:
 * - Coordinates strategy selection, execution, and cross-validation
 * - Provides simple API for verification requests
 * - Handles storage and learning feedback loops
 * - Follows idempotency and timeout laws
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnifiedVerificationOrchestrator = void 0;
const uuid_1 = require("uuid");
const strategy_selector_1 = require("./strategy-selector");
const cross_validator_1 = require("./cross-validator");
const confidence_aggregator_1 = require("./confidence-aggregator");
const logger_1 = require("../../lib/logger");
/**
 * Main Orchestrator - coordinates all verification components
 */
class UnifiedVerificationOrchestrator {
    constructor(z3Url, leanaideUrl, logger) {
        // Storage for learning (would connect to Vector DB + Graphiti)
        this.resultHistory = new Map();
        this.logger = logger || new logger_1.Logger('UnifiedVerificationOrchestrator');
        // Validate configuration
        if (!z3Url) {
            throw new Error('Z3_URL environment variable is required');
        }
        if (!leanaideUrl) {
            throw new Error('LEANAIDE_URL environment variable is required');
        }
        // Initialize components
        this.strategySelector = new strategy_selector_1.VerificationStrategySelector(this.logger);
        this.crossValidator = new cross_validator_1.CrossValidator(z3Url, leanaideUrl, this.logger);
        this.confidenceAggregator = new confidence_aggregator_1.ConfidenceAggregator(this.logger);
        this.logger.info({
            msg: 'UnifiedVerificationOrchestrator initialized',
            z3Url,
            leanaideUrl
        });
    }
    /**
     * Simple verification - single system or basic approach
     */
    async verify(problem, constraints, options = {}) {
        const correlationId = options.correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Starting verification',
            correlationId,
            problemId: problem.id,
            problemType: problem.type
        });
        try {
            // Build request
            const request = {
                requestId: (0, uuid_1.v4)(),
                problem,
                constraints,
                strategy: options.strategy,
                confidenceRequired: options.confidenceRequired,
                timestamp: new Date().toISOString(),
                correlationId
            };
            // Select strategy if not provided
            if (!options.strategy) {
                const selection = await this.strategySelector.selectStrategy(request);
                request.strategy = selection.strategy;
            }
            // Execute verification
            const systemResults = await this.crossValidator.executeVerification(request, request.strategy);
            // Get primary result (first one)
            const primaryResult = systemResults[0];
            const result = {
                system: primaryResult.system,
                verified: primaryResult.verified,
                confidence: primaryResult.confidence,
                output: primaryResult.output,
                proof: primaryResult.proof,
                metadata: {
                    executionTime: primaryResult.executionTime,
                    memoryUsed: primaryResult.memoryUsed,
                    strategy: request.strategy,
                    timestamp: primaryResult.timestamp,
                    errorMessage: primaryResult.errorMessage
                }
            };
            // Store if requested
            if (options.storeResults !== false) {
                await this.storeResults(result, correlationId);
            }
            // Learn from outcome
            await this.learnFromOutcome(result, problem);
            this.logger.info({
                msg: 'Verification completed',
                correlationId,
                verified: result.verified,
                confidence: result.confidence
            });
            return result;
        }
        catch (error) {
            this.logger.error({
                msg: 'Verification failed',
                correlationId,
                error: error instanceof Error ? error.message : 'Unknown error',
                stack: error instanceof Error ? error.stack : undefined
            });
            throw error;
        }
    }
    /**
     * Verification with cross-validation - multiple systems
     */
    async verifyWithCrossValidation(problem, options = {}) {
        const correlationId = options.correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Starting cross-validation verification',
            correlationId,
            problemId: problem.id,
            problemType: problem.type
        });
        try {
            // Build request
            const request = {
                requestId: (0, uuid_1.v4)(),
                problem,
                constraints: {
                    timeout: options.timeout,
                    precision: 'high',
                    allowedSystems: ['both'],
                    requiredConfidence: options.confidenceRequired
                },
                strategy: options.strategy,
                confidenceRequired: options.confidenceRequired,
                timestamp: new Date().toISOString(),
                correlationId
            };
            // Select strategy if not provided
            if (!options.strategy) {
                const selection = await this.strategySelector.selectStrategy(request);
                request.strategy = selection.strategy;
            }
            // Execute cross-validation
            const crossValidationResult = await this.crossValidator.validate(request);
            // Aggregate confidence
            const systemResults = this.crossValidatorToSystemResults(crossValidationResult);
            const confidenceScore = await this.confidenceAggregator.aggregate(systemResults, crossValidationResult.strategy, options.confidenceRequired);
            // Update result with aggregated confidence
            crossValidationResult.confidence = confidenceScore.combined;
            // Store if requested
            if (options.storeResults !== false) {
                await this.storeCrossValidationResults(crossValidationResult, correlationId);
            }
            // Learn from outcomes
            for (const result of crossValidationResult.systemResults) {
                await this.learnFromOutcome(result, problem);
            }
            this.logger.info({
                msg: 'Cross-validation verification completed',
                correlationId,
                verified: crossValidationResult.verified,
                agreement: crossValidationResult.agreement,
                confidence: crossValidationResult.confidence,
                resolution: crossValidationResult.resolution
            });
            return crossValidationResult;
        }
        catch (error) {
            this.logger.error({
                msg: 'Cross-validation verification failed',
                correlationId,
                error: error instanceof Error ? error.message : 'Unknown error',
                stack: error instanceof Error ? error.stack : undefined
            });
            throw error;
        }
    }
    /**
     * Batch verification - multiple problems
     */
    async verifyBatch(problems, constraints, options = {}) {
        const correlationId = options.correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Starting batch verification',
            correlationId,
            problemCount: problems.length
        });
        const results = new Map();
        // Process in parallel with concurrency limit
        const concurrencyLimit = 5;
        const batches = [];
        for (let i = 0; i < problems.length; i += concurrencyLimit) {
            batches.push(problems.slice(i, i + concurrencyLimit));
        }
        for (const batch of batches) {
            const batchResults = await Promise.all(batch.map(problem => this.verify(problem, constraints, { ...options, correlationId })
                .catch(error => {
                this.logger.error({
                    msg: 'Batch item failed',
                    correlationId,
                    problemId: problem.id,
                    error: error instanceof Error ? error.message : 'Unknown error'
                });
                return null;
            })));
            batchResults.forEach((result, index) => {
                if (result) {
                    results.set(batch[index].id, result);
                }
            });
        }
        this.logger.info({
            msg: 'Batch verification completed',
            correlationId,
            total: problems.length,
            successful: results.size,
            failed: problems.length - results.size
        });
        return results;
    }
    /**
     * Store verification result for learning
     */
    async storeResults(result, correlationId) {
        try {
            const key = `${result.system}_${correlationId}`;
            if (!this.resultHistory.has(key)) {
                this.resultHistory.set(key, []);
            }
            this.resultHistory.get(key).push(result);
            this.logger.debug({
                msg: 'Result stored',
                correlationId,
                system: result.system,
                verified: result.verified
            });
            // TODO: Store in Vector DB + Graphiti
            // await this.vectorStore.store(result);
            // await this.graphiti.storeRelationship(result);
        }
        catch (error) {
            this.logger.error({
                msg: 'Failed to store result',
                correlationId,
                error: error instanceof Error ? error.message : 'Unknown error'
            });
            // Don't throw - storage failure shouldn't break verification
        }
    }
    /**
     * Store cross-validation result for learning
     */
    async storeCrossValidationResults(result, correlationId) {
        try {
            // Store each system result
            for (const systemResult of result.systemResults) {
                await this.storeResults(systemResult, correlationId);
            }
            this.logger.debug({
                msg: 'Cross-validation results stored',
                correlationId,
                resolution: result.resolution
            });
            // TODO: Store cross-validation metadata in Vector DB + Graphiti
        }
        catch (error) {
            this.logger.error({
                msg: 'Failed to store cross-validation results',
                correlationId,
                error: error instanceof Error ? error.message : 'Unknown error'
            });
        }
    }
    /**
     * Learn from verification outcomes to improve strategy selection
     */
    async learnFromOutcome(result, problem) {
        try {
            // Update strategy effectiveness
            await this.strategySelector.updateEffectiveness(result.metadata.strategy, problem.type, result.verified, result.metadata.executionTime, result.confidence);
            // Update confidence aggregator accuracy
            await this.confidenceAggregator.updateAccuracy(result.system, result.confidence, result.verified);
            this.logger.debug({
                msg: 'Learning from outcome',
                system: result.system,
                strategy: result.metadata.strategy,
                verified: result.verified,
                confidence: result.confidence
            });
        }
        catch (error) {
            this.logger.error({
                msg: 'Failed to learn from outcome',
                error: error instanceof Error ? error.message : 'Unknown error'
            });
        }
    }
    /**
     * Get statistics about verification performance
     */
    async getStatistics() {
        let totalVerifications = 0;
        let successful = 0;
        let totalConfidence = 0;
        let totalExecutionTime = 0;
        let z3Count = 0;
        let z3Successful = 0;
        let leanaideCount = 0;
        let leanaideSuccessful = 0;
        for (const [, results] of this.resultHistory) {
            for (const result of results) {
                totalVerifications++;
                if (result.verified)
                    successful++;
                totalConfidence += result.confidence;
                totalExecutionTime += result.metadata.executionTime;
                if (result.system === 'z3') {
                    z3Count++;
                    if (result.verified)
                        z3Successful++;
                }
                else if (result.system === 'leanaide') {
                    leanaideCount++;
                    if (result.verified)
                        leanaideSuccessful++;
                }
            }
        }
        return {
            totalVerifications,
            successRate: totalVerifications > 0 ? successful / totalVerifications : 0,
            averageConfidence: totalVerifications > 0 ? totalConfidence / totalVerifications : 0,
            averageExecutionTime: totalVerifications > 0 ? totalExecutionTime / totalVerifications : 0,
            systemBreakdown: {
                z3: {
                    count: z3Count,
                    successRate: z3Count > 0 ? z3Successful / z3Count : 0
                },
                leanaide: {
                    count: leanaideCount,
                    successRate: leanaideCount > 0 ? leanaideSuccessful / leanaideCount : 0
                }
            }
        };
    }
    /**
     * Health check for the orchestrator
     */
    async healthCheck() {
        return {
            healthy: true,
            components: {
                orchestrator: true,
                strategySelector: true,
                crossValidator: true,
                confidenceAggregator: true
            }
        };
    }
    /**
     * Helper: Convert CrossValidationResult to SystemResult[]
     */
    crossValidatorToSystemResults(result) {
        return result.systemResults.map(r => ({
            system: r.system,
            verified: r.verified,
            confidence: r.confidence,
            output: r.output,
            proof: r.proof,
            executionTime: r.metadata.executionTime,
            memoryUsed: r.metadata.memoryUsed,
            errorMessage: r.metadata.errorMessage,
            timestamp: r.metadata.timestamp
        }));
    }
}
exports.UnifiedVerificationOrchestrator = UnifiedVerificationOrchestrator;
//# sourceMappingURL=orchestrator.js.map