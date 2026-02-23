"use strict";
/**
 * CROSS VALIDATOR
 *
 * Executes verification across multiple systems and validates results:
 * - Parallel execution: Run both systems simultaneously
 * - Sequential execution: Run one, then the other if needed
 * - Compare results and detect disagreements
 * - Resolve conflicts based on confidence and evidence
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CrossValidator = void 0;
const uuid_1 = require("uuid");
const axios_1 = __importDefault(require("axios"));
const logger_1 = require("../../lib/logger");
/**
 * Cross Validator - orchestrates multi-system verification
 */
class CrossValidator {
    constructor(z3Url, leanaideUrl, logger) {
        this.systems = new Map();
        this.logger = logger || new logger_1.Logger('CrossValidator');
        // Validate environment variables (Law of Configuration Explicitness)
        if (!z3Url) {
            throw new Error('Z3_URL environment variable is required');
        }
        if (!leanaideUrl) {
            throw new Error('LEANAIDE_URL environment variable is required');
        }
        // Configure system endpoints
        this.systems.set('z3', {
            name: 'z3',
            url: z3Url,
            timeout: parseInt(process.env.Z3_TIMEOUT || '30000', 10),
            healthCheckPath: process.env.Z3_HEALTH_CHECK || '/health',
            verifyPath: process.env.Z3_VERIFY_PATH || '/verify'
        });
        this.systems.set('leanaide', {
            name: 'leanaide',
            url: leanaideUrl,
            timeout: parseInt(process.env.LEANAIDE_TIMEOUT || '45000', 10),
            healthCheckPath: process.env.LEANAIDE_HEALTH_CHECK || '/health',
            verifyPath: process.env.LEANAIDE_VERIFY_PATH || '/verify'
        });
        this.logger.info({
            msg: 'CrossValidator initialized',
            z3Url,
            leanaideUrl
        });
    }
    /**
     * Main entry point - validate with cross-checking
     */
    async validate(request) {
        const correlationId = request.correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Starting cross-validation',
            correlationId,
            requestId: request.requestId,
            problemType: request.problem.type
        });
        const startTime = Date.now();
        try {
            // Determine execution approach
            const strategy = request.strategy || this.inferStrategy(request);
            // Execute verification(s)
            const systemResults = await this.executeVerification(request, strategy);
            // Compare results
            const comparison = this.compareResults(systemResults);
            // Detect disagreements
            const conflicts = this.detectDisagreements(systemResults, comparison);
            // Determine resolution
            const resolution = this.determineResolution(systemResults, conflicts, comparison);
            const totalTime = Date.now() - startTime;
            const result = {
                requestId: request.requestId,
                verified: resolution === 'verified',
                agreement: comparison.agreement,
                agreementType: comparison.agreementType,
                confidence: this.calculateAggregateConfidence(systemResults),
                systemResults: this.toVerificationResults(systemResults),
                conflicts,
                resolution,
                strategy,
                metadata: {
                    correlationId,
                    totalExecutionTime: totalTime,
                    timestamp: new Date().toISOString()
                }
            };
            this.logger.info({
                msg: 'Cross-validation completed',
                correlationId,
                verified: result.verified,
                agreement: result.agreement,
                confidence: result.confidence,
                executionTime: totalTime
            });
            return result;
        }
        catch (error) {
            this.logger.error({
                msg: 'Cross-validation failed',
                correlationId,
                error: error instanceof Error ? error.message : 'Unknown error',
                stack: error instanceof Error ? error.stack : undefined
            });
            throw error;
        }
    }
    /**
     * Execute verification based on strategy
     */
    async executeVerification(request, strategy) {
        switch (strategy) {
            case 'z3_only':
                return [await this.executeZ3(request)];
            case 'leanaide_only':
                return [await this.executeLeanAide(request)];
            case 'parallel':
                return await this.executeParallel(request);
            case 'sequential':
                return await this.executeSequential(request);
            case 'hybrid':
                return await this.executeHybrid(request);
            default:
                throw new Error(`Unknown strategy: ${strategy}`);
        }
    }
    /**
     * Execute Z3 verification
     */
    async executeZ3(request) {
        const system = this.systems.get('z3');
        if (!system) {
            throw new Error('Z3 system not configured');
        }
        this.logger.info({
            msg: 'Executing Z3 verification',
            correlationId: request.correlationId,
            problemId: request.problem.id
        });
        const startTime = Date.now();
        try {
            // Circuit breaker check (TODO: implement circuit breaker)
            // const isHealthy = await this.checkSystemHealth(system);
            // if (!isHealthy) {
            //   throw new Error('Z3 system is unhealthy');
            // }
            const response = await axios_1.default.post(`${system.url}${system.verifyPath}`, {
                problem: request.problem,
                constraints: {
                    timeout: system.timeout,
                    precision: request.constraints.precision
                }
            }, {
                timeout: system.timeout,
                headers: {
                    'Content-Type': 'application/json',
                    'X-Correlation-ID': request.correlationId
                }
            });
            const executionTime = Date.now() - startTime;
            // Normalize to canonical format
            const result = {
                system: 'z3',
                verified: response.data.verified || false,
                confidence: response.data.confidence || 0.0,
                output: response.data.output || '',
                proof: response.data.proof,
                executionTime,
                memoryUsed: response.data.memoryUsed,
                timestamp: new Date().toISOString()
            };
            this.logger.info({
                msg: 'Z3 verification completed',
                correlationId: request.correlationId,
                verified: result.verified,
                confidence: result.confidence,
                executionTime
            });
            return result;
        }
        catch (error) {
            const executionTime = Date.now() - startTime;
            if (axios_1.default.isAxiosError(error)) {
                const axiosError = error;
                this.logger.error({
                    msg: 'Z3 verification request failed',
                    correlationId: request.correlationId,
                    status: axiosError.response?.status,
                    statusText: axiosError.response?.statusText,
                    data: axiosError.response?.data
                });
                // Return failure result (don't throw - handle gracefully)
                return {
                    system: 'z3',
                    verified: false,
                    confidence: 0.0,
                    output: '',
                    executionTime,
                    errorMessage: `HTTP ${axiosError.response?.status}: ${axiosError.message}`,
                    timestamp: new Date().toISOString()
                };
            }
            this.logger.error({
                msg: 'Z3 verification error',
                correlationId: request.correlationId,
                error: error instanceof Error ? error.message : 'Unknown error'
            });
            return {
                system: 'z3',
                verified: false,
                confidence: 0.0,
                output: '',
                executionTime,
                errorMessage: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString()
            };
        }
    }
    /**
     * Execute LeanAide verification
     */
    async executeLeanAide(request) {
        const system = this.systems.get('leanaide');
        if (!system) {
            throw new Error('LeanAide system not configured');
        }
        this.logger.info({
            msg: 'Executing LeanAide verification',
            correlationId: request.correlationId,
            problemId: request.problem.id
        });
        const startTime = Date.now();
        try {
            const response = await axios_1.default.post(`${system.url}${system.verifyPath}`, {
                problem: request.problem,
                constraints: {
                    timeout: system.timeout,
                    precision: request.constraints.precision
                }
            }, {
                timeout: system.timeout,
                headers: {
                    'Content-Type': 'application/json',
                    'X-Correlation-ID': request.correlationId
                }
            });
            const executionTime = Date.now() - startTime;
            const result = {
                system: 'leanaide',
                verified: response.data.verified || false,
                confidence: response.data.confidence || 0.0,
                output: response.data.output || '',
                proof: response.data.proof,
                executionTime,
                memoryUsed: response.data.memoryUsed,
                timestamp: new Date().toISOString()
            };
            this.logger.info({
                msg: 'LeanAide verification completed',
                correlationId: request.correlationId,
                verified: result.verified,
                confidence: result.confidence,
                executionTime
            });
            return result;
        }
        catch (error) {
            const executionTime = Date.now() - startTime;
            if (axios_1.default.isAxiosError(error)) {
                const axiosError = error;
                this.logger.error({
                    msg: 'LeanAide verification request failed',
                    correlationId: request.correlationId,
                    status: axiosError.response?.status,
                    statusText: axiosError.response?.statusText
                });
                return {
                    system: 'leanaide',
                    verified: false,
                    confidence: 0.0,
                    output: '',
                    executionTime,
                    errorMessage: `HTTP ${axiosError.response?.status}: ${axiosError.message}`,
                    timestamp: new Date().toISOString()
                };
            }
            this.logger.error({
                msg: 'LeanAide verification error',
                correlationId: request.correlationId,
                error: error instanceof Error ? error.message : 'Unknown error'
            });
            return {
                system: 'leanaide',
                verified: false,
                confidence: 0.0,
                output: '',
                executionTime,
                errorMessage: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString()
            };
        }
    }
    /**
     * Execute both systems in parallel
     */
    async executeParallel(request) {
        this.logger.info({
            msg: 'Executing parallel verification',
            correlationId: request.correlationId
        });
        const results = await Promise.all([
            this.executeZ3(request),
            this.executeLeanAide(request)
        ]);
        return results;
    }
    /**
     * Execute systems sequentially (stop if first succeeds)
     */
    async executeSequential(request) {
        this.logger.info({
            msg: 'Executing sequential verification',
            correlationId: request.correlationId
        });
        const results = [];
        // Try Z3 first
        const z3Result = await this.executeZ3(request);
        results.push(z3Result);
        // If Z3 succeeded with high confidence, skip LeanAide
        if (z3Result.verified && z3Result.confidence >= request.confidenceRequired) {
            this.logger.info({
                msg: 'Z3 verified with sufficient confidence, skipping LeanAide',
                correlationId: request.correlationId,
                confidence: z3Result.confidence
            });
            return results;
        }
        // Try LeanAide
        const leanaideResult = await this.executeLeanAide(request);
        results.push(leanaideResult);
        return results;
    }
    /**
     * Execute hybrid approach (Z3 first, then LeanAide for proof refinement)
     */
    async executeHybrid(request) {
        this.logger.info({
            msg: 'Executing hybrid verification',
            correlationId: request.correlationId
        });
        // First pass: Z3 for quick verification
        const z3Result = await this.executeZ3(request);
        // If Z3 fails, use LeanAide for deeper analysis
        if (!z3Result.verified) {
            this.logger.info({
                msg: 'Z3 verification failed, trying LeanAide',
                correlationId: request.correlationId
            });
            const leanaideResult = await this.executeLeanAide(request);
            return [z3Result, leanaideResult];
        }
        // If Z3 succeeds but confidence is low, cross-validate with LeanAide
        if (z3Result.confidence < request.confidenceRequired) {
            this.logger.info({
                msg: 'Z3 confidence low, cross-validating with LeanAide',
                correlationId: request.correlationId,
                z3Confidence: z3Result.confidence
            });
            const leanaideResult = await this.executeLeanAide(request);
            return [z3Result, leanaideResult];
        }
        // Z3 succeeded with high confidence
        return [z3Result];
    }
    /**
     * Compare results from multiple systems
     */
    compareResults(results) {
        if (results.length < 2) {
            return {
                agreement: true,
                agreementType: 'full_agreement',
                confidenceAlignment: 1.0,
                verificationAlignment: true,
                details: 'Single system result'
            };
        }
        const [r1, r2] = results;
        // Check if verification outcomes agree
        const verificationAlignment = r1.verified === r2.verified;
        // Check confidence alignment
        const confidenceDiff = Math.abs(r1.confidence - r2.confidence);
        const confidenceAlignment = 1 - confidenceDiff;
        // Determine agreement type
        let agreementType;
        let agreement;
        if (verificationAlignment && confidenceAlignment > 0.9) {
            agreementType = 'full_agreement';
            agreement = true;
        }
        else if (verificationAlignment && confidenceAlignment > 0.7) {
            agreementType = 'partial_agreement';
            agreement = true;
        }
        else if (!verificationAlignment && confidenceAlignment > 0.7) {
            agreementType = 'disagreement';
            agreement = false;
        }
        else {
            agreementType = 'inconclusive';
            agreement = false;
        }
        const details = this.generateComparisonDetails(results, agreementType);
        return {
            agreement,
            agreementType,
            confidenceAlignment,
            verificationAlignment,
            details
        };
    }
    /**
     * Detect disagreements between results
     */
    detectDisagreements(results, comparison) {
        const disagreements = [];
        if (results.length < 2) {
            return disagreements;
        }
        const [r1, r2] = results;
        // Verification outcome disagreement
        if (!comparison.verificationAlignment) {
            disagreements.push({
                type: 'verification_outcome',
                systemA: r1.system,
                systemB: r2.system,
                description: `Systems disagree on verification outcome: ${r1.system} says ${r1.verified}, ${r2.system} says ${r2.verified}`,
                severity: 'critical',
                resolution: 'trust_higher_confidence'
            });
        }
        // Confidence level disagreement
        const confidenceDiff = Math.abs(r1.confidence - r2.confidence);
        if (confidenceDiff > 0.3) {
            disagreements.push({
                type: 'confidence_level',
                systemA: r1.system,
                systemB: r2.system,
                description: `Large confidence gap: ${r1.system}=${r1.confidence}, ${r2.system}=${r2.confidence}`,
                severity: confidenceDiff > 0.5 ? 'high' : 'medium',
                resolution: 'trust_higher_confidence'
            });
        }
        return disagreements;
    }
    /**
     * Determine final resolution based on results and conflicts
     */
    determineResolution(results, conflicts, comparison) {
        // If any system has critical error
        const hasCriticalErrors = results.some(r => r.errorMessage !== undefined && !r.verified);
        if (hasCriticalErrors && results.every(r => !r.verified)) {
            return 'not_verified';
        }
        // Full agreement with high confidence
        if (comparison.agreementType === 'full_agreement') {
            const avgConfidence = this.calculateAggregateConfidence(results);
            return avgConfidence >= 0.9 ? 'verified' : 'verified';
        }
        // Critical disagreements
        const criticalConflicts = conflicts.filter(c => c.severity === 'critical');
        if (criticalConflicts.length > 0) {
            return 'requires_review';
        }
        // Partial agreement - check if we can trust higher confidence
        if (comparison.agreementType === 'partial_agreement') {
            const highest = results.reduce((prev, current) => prev.confidence > current.confidence ? prev : current);
            return highest.verified ? 'verified' : 'not_verified';
        }
        // Inconclusive
        if (comparison.agreementType === 'inconclusive') {
            return 'requires_review';
        }
        // Default
        return 'inconclusive';
    }
    /**
     * Calculate aggregate confidence from multiple results
     */
    calculateAggregateConfidence(results) {
        if (results.length === 0)
            return 0;
        if (results.length === 1)
            return results[0].confidence;
        // Weighted average favoring higher confidence
        const weights = results.map(r => r.confidence);
        const totalWeight = weights.reduce((sum, w) => sum + w, 0);
        const weightedSum = results.reduce((sum, r, i) => {
            return sum + (r.confidence * weights[i]);
        }, 0);
        return weightedSum / totalWeight;
    }
    /**
     * Infer strategy from request if not specified
     */
    inferStrategy(request) {
        const allowed = request.constraints.allowedSystems;
        if (allowed.includes('z3') && !allowed.includes('leanaide')) {
            return 'z3_only';
        }
        if (allowed.includes('leanaide') && !allowed.includes('z3')) {
            return 'leanaide_only';
        }
        return 'parallel';
    }
    /**
     * Convert SystemResult[] to VerificationResult[]
     */
    toVerificationResults(results) {
        return results.map(r => ({
            system: r.system,
            verified: r.verified,
            confidence: r.confidence,
            output: r.output,
            proof: r.proof,
            metadata: {
                executionTime: r.executionTime,
                memoryUsed: r.memoryUsed,
                strategy: 'parallel', // Will be set by caller
                timestamp: r.timestamp,
                errorMessage: r.errorMessage
            }
        }));
    }
    /**
     * Generate human-readable comparison details
     */
    generateComparisonDetails(results, agreementType) {
        const [r1, r2] = results;
        if (results.length < 2) {
            return 'Single system execution';
        }
        const details = {
            'full_agreement': `Both ${r1.system} and ${r2.system} agree: verified=${r1.verified}, confidence alignment=${(Math.abs(r1.confidence - r2.confidence) < 0.1 ? 'excellent' : 'good')}`,
            'partial_agreement': `Systems agree on verification outcome=${r1.verified} but have some confidence variance`,
            'disagreement': `Systems disagree on verification outcome: ${r1.system}=${r1.verified}, ${r2.system}=${r2.verified}`,
            'inconclusive': `Results are inconclusive due to significant discrepancies`
        };
        return details[agreementType];
    }
    /**
     * Check system health (circuit breaker prerequisite)
     */
    async checkSystemHealth(system) {
        try {
            const response = await axios_1.default.get(`${system.url}${system.healthCheckPath}`, { timeout: 5000 });
            return response.status === 200;
        }
        catch {
            return false;
        }
    }
}
exports.CrossValidator = CrossValidator;
//# sourceMappingURL=cross-validator.js.map