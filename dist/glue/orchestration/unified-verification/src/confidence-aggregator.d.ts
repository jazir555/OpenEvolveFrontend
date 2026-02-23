/**
 * CONFIDENCE AGGREGATOR
 *
 * Combines confidence scores from multiple verification systems:
 * - Normalizes scores to common scale
 * - Applies dynamic weights based on system effectiveness
 * - Generates evidence trail for explainability
 * - Determines if threshold is met
 */
import { SystemResult, ConfidenceScore, VerificationStrategy } from './canonical';
import { Logger } from '../../lib/logger';
export interface NormalizedScore {
    system: 'z3' | 'leanaide';
    originalScore: number;
    normalizedScore: number;
    weight: number;
    factors: {
        historicalAccuracy: number;
        problemTypeMatch: number;
        executionQuality: number;
        confidenceConsistency: number;
    };
}
export interface WeightMap {
    z3: number;
    leanaide: number;
}
export interface CombinedScore {
    score: number;
    confidence: 'low' | 'medium' | 'high' | 'very_high';
    metThreshold: boolean;
    explanation: string;
}
export interface Evidence {
    source: string;
    weight: number;
    description: string;
    data?: Record<string, unknown>;
}
/**
 * Confidence Aggregator - combines and normalizes confidence scores
 */
export declare class ConfidenceAggregator {
    private logger;
    private accuracyHistory;
    constructor(logger?: Logger);
    /**
     * Main entry point - aggregate confidence from multiple results
     */
    aggregate(results: SystemResult[], strategy: VerificationStrategy, requiredThreshold?: number): Promise<ConfidenceScore>;
    /**
     * Normalize scores to common scale
     */
    private normalizeScores;
    /**
     * Calculate factors for normalization
     */
    private calculateNormalizationFactors;
    /**
     * Calculate dynamic weights for each system
     */
    private calculateWeights;
    /**
     * Combine normalized scores using weights
     */
    private combineScores;
    /**
     * Generate evidence trail
     */
    private generateEvidence;
    /**
     * Generate human-readable score explanation
     */
    private generateScoreExplanation;
    /**
     * Load accuracy history from learning data
     */
    private loadAccuracyHistory;
    /**
     * Update accuracy history based on outcomes
     */
    updateAccuracy(system: 'z3' | 'leanaide', predictedConfidence: number, actualCorrect: boolean): Promise<void>;
    /**
     * Get current accuracy for a system
     */
    getSystemAccuracy(system: 'z3' | 'leanaide'): number;
}
//# sourceMappingURL=confidence-aggregator.d.ts.map