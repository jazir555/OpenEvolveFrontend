/**
 * CONFIDENCE AGGREGATOR
 *
 * Combines confidence scores from multiple verification systems:
 * - Normalizes scores to common scale
 * - Applies dynamic weights based on system effectiveness
 * - Generates evidence trail for explainability
 * - Determines if threshold is met
 */

import { v4 as uuidv4 } from 'uuid';
import {
  SystemResult,
  ConfidenceScore,
  VerificationStrategy,
  VerificationResult
} from './canonical';
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
export class ConfidenceAggregator {
  private logger: Logger;
  private accuracyHistory: Map<string, number> = new Map();

  constructor(logger?: Logger) {
    this.logger = logger || new Logger('ConfidenceAggregator');
    this.loadAccuracyHistory();
  }

  /**
   * Main entry point - aggregate confidence from multiple results
   */
  async aggregate(
    results: SystemResult[],
    strategy: VerificationStrategy,
    requiredThreshold: number = 0.95
  ): Promise<ConfidenceScore> {
    const correlationId = uuidv4();

    this.logger.info({
      msg: 'Aggregating confidence scores',
      correlationId,
      resultCount: results.length,
      strategy,
      requiredThreshold
    });

    try {
      // Normalize individual scores
      const normalized = this.normalizeScores(results);

      // Calculate dynamic weights
      const weights = this.calculateWeights(results, strategy);

      // Combine scores
      const combined = this.combineScores(normalized, weights);

      // Generate evidence
      const evidence = this.generateEvidence(results, normalized, weights);

      const individual: Record<string, number> = {};
      results.forEach(r => {
        individual[r.system] = r.confidence;
      });

      const score: ConfidenceScore = {
        combined: combined.score,
        individual,
        weights,
        evidence,
        meetsThreshold: combined.metThreshold,
        timestamp: new Date().toISOString()
      };

      this.logger.info({
        msg: 'Confidence aggregation completed',
        correlationId,
        combined: score.combined,
        meetsThreshold: score.meetsThreshold,
        evidenceCount: evidence.length
      });

      return score;
    } catch (error) {
      this.logger.error({
        msg: 'Confidence aggregation failed',
        correlationId,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      throw error;
    }
  }

  /**
   * Normalize scores to common scale
   */
  private normalizeScores(results: SystemResult[]): NormalizedScore[] {
    return results.map(result => {
      const factors = this.calculateNormalizationFactors(result);

      // Base normalization
      let normalized = result.confidence;

      // Adjust based on factors
      normalized *= (
        factors.historicalAccuracy * 0.3
        + factors.problemTypeMatch * 0.3
        + factors.executionQuality * 0.2
        + factors.confidenceConsistency * 0.2
      );

      // Ensure within [0, 1]
      normalized = Math.max(0, Math.min(1, normalized));

      return {
        system: result.system,
        originalScore: result.confidence,
        normalizedScore: normalized,
        weight: 0, // Will be calculated in calculateWeights
        factors
      };
    });
  }

  /**
   * Calculate factors for normalization
   */
  private calculateNormalizationFactors(result: SystemResult): NormalizedScore['factors'] {
    // Historical accuracy (from learning data)
    const accuracyKey = `${result.system}_accuracy`;
    const historicalAccuracy = this.accuracyHistory.get(accuracyKey) || 0.85;

    // Execution quality (based on execution time and errors)
    let executionQuality = 1.0;
    if (result.errorMessage) {
      executionQuality = 0.3;
    } else if (result.executionTime > 30000) {
      executionQuality = 0.7;
    } else if (result.executionTime > 10000) {
      executionQuality = 0.9;
    }

    // Confidence consistency (penalty for extreme values)
    const confidenceConsistency = result.confidence > 0.1 && result.confidence < 0.95
      ? 1.0
      : 0.8;

    // Problem type match (simplified - would use actual problem type matching)
    const problemTypeMatch = 0.9;

    return {
      historicalAccuracy,
      problemTypeMatch,
      executionQuality,
      confidenceConsistency
    };
  }

  /**
   * Calculate dynamic weights for each system
   */
  private calculateWeights(
    results: SystemResult[],
    strategy: VerificationStrategy
  ): WeightMap {
    const weights: WeightMap = {
      z3: 0,
      leanaide: 0
    };

    // Base weights by strategy
    const baseWeights: Record<VerificationStrategy, { z3: number; leanaide: number }> = {
      'z3_only': { z3: 1.0, leanaide: 0.0 },
      'leanaide_only': { z3: 0.0, leanaide: 1.0 },
      'parallel': { z3: 0.5, leanaide: 0.5 },
      'sequential': { z3: 0.6, leanaide: 0.4 },
      'hybrid': { z3: 0.55, leanaide: 0.45 }
    };

    const base = baseWeights[strategy];

    // Adjust based on actual results
    results.forEach(result => {
      let weight = base[result.system];

      // Boost weight for successful verification
      if (result.verified) {
        weight *= 1.2;
      }

      // Reduce weight for errors
      if (result.errorMessage) {
        weight *= 0.3;
      }

      // Adjust based on confidence
      weight *= (0.5 + result.confidence);

      weights[result.system] = weight;
    });

    // Normalize weights to sum to 1
    const total = weights.z3 + weights.leanaide;
    if (total > 0) {
      weights.z3 /= total;
      weights.leanaide /= total;
    } else {
      // Equal weights if both failed
      weights.z3 = 0.5;
      weights.leanaide = 0.5;
    }

    return weights;
  }

  /**
   * Combine normalized scores using weights
   */
  private combineScores(
    normalized: NormalizedScore[],
    weights: WeightMap
  ): CombinedScore {
    let combined = 0;

    normalized.forEach(n => {
      combined += n.normalizedScore * weights[n.system];
    });

    // Ensure within [0, 1]
    combined = Math.max(0, Math.min(1, combined));

    // Determine confidence level
    let confidence: CombinedScore['confidence'];
    if (combined >= 0.95) {
      confidence = 'very_high';
    } else if (combined >= 0.85) {
      confidence = 'high';
    } else if (combined >= 0.7) {
      confidence = 'medium';
    } else {
      confidence = 'low';
    }

    // Generate explanation
    const explanation = this.generateScoreExplanation(normalized, weights, combined);

    return {
      score: combined,
      confidence,
      metThreshold: combined >= 0.95, // Default threshold
      explanation
    };
  }

  /**
   * Generate evidence trail
   */
  private generateEvidence(
    results: SystemResult[],
    normalized: NormalizedScore[],
    weights: WeightMap
  ): Evidence[] {
    const evidence: Evidence[] = [];

    // Evidence for each system
    results.forEach(result => {
      const norm = normalized.find(n => n.system === result.system);
      if (!norm) return;

      evidence.push({
        source: `${result.system}_verification`,
        weight: weights[result.system],
        description: `${result.system} verification result: ${result.verified ? 'VERIFIED' : 'NOT VERIFIED'} with confidence ${result.confidence}`,
        data: {
          verified: result.verified,
          confidence: result.confidence,
          normalized: norm.normalizedScore,
          executionTime: result.executionTime,
          hasProof: !!result.proof,
          hasError: !!result.errorMessage
        }
      });

      // Evidence for normalization factors
      evidence.push({
        source: `${result.system}_normalization`,
        weight: 0.1,
        description: `${result.system} score adjusted based on historical accuracy, problem type match, and execution quality`,
        data: {
          originalScore: result.confidence,
          normalizedScore: norm.normalizedScore,
          factors: norm.factors
        }
      });
    });

    // Agreement evidence (if multiple systems)
    if (results.length > 1) {
      const [r1, r2] = results;
      const agreement = r1.verified === r2.verified;
      const confidenceDiff = Math.abs(r1.confidence - r2.confidence);

      evidence.push({
        source: 'cross_validation',
        weight: 0.2,
        description: `Cross-validation: ${agreement ? 'AGREEMENT' : 'DISAGREEMENT'} between systems`,
        data: {
          agreement,
          confidenceVariance: confidenceDiff,
          bothVerified: r1.verified && r2.verified,
          bothFailed: !r1.verified && !r2.verified
        }
      });
    }

    return evidence;
  }

  /**
   * Generate human-readable score explanation
   */
  private generateScoreExplanation(
    normalized: NormalizedScore[],
    weights: WeightMap,
    combined: number
  ): string {
    const parts: string[] = [];

    normalized.forEach(n => {
      parts.push(
        `${n.system}: ${(n.normalizedScore * 100).toFixed(1)}% (weight: ${(weights[n.system] * 100).toFixed(1)}%)`
      );
    });

    const breakdown = parts.join(', ');
    const total = `Combined: ${(combined * 100).toFixed(1)}%`;

    return `${total}. Breakdown: ${breakdown}`;
  }

  /**
   * Load accuracy history from learning data
   */
  private loadAccuracyHistory(): void {
    // TODO: Load from Vector DB/Graphiti
    // Default values for now
    this.accuracyHistory.set('z3_accuracy', 0.90);
    this.accuracyHistory.set('leanaide_accuracy', 0.88);

    this.logger.info({
      msg: 'Accuracy history loaded',
      entries: this.accuracyHistory.size
    });
  }

  /**
   * Update accuracy history based on outcomes
   */
  async updateAccuracy(
    system: 'z3' | 'leanaide',
    predictedConfidence: number,
    actualCorrect: boolean
  ): Promise<void> {
    const key = `${system}_accuracy`;
    const existing = this.accuracyHistory.get(key) || 0.85;

    // Update with exponential moving average
    const alpha = 0.1;
    const newAccuracy = alpha * (actualCorrect ? 1.0 : 0.0) + (1 - alpha) * existing;

    this.accuracyHistory.set(key, newAccuracy);

    this.logger.info({
      msg: 'Accuracy history updated',
      system,
      previousAccuracy: existing,
      newAccuracy,
      predictedConfidence,
      actualCorrect
    });

    // TODO: Persist to Vector DB/Graphiti
  }

  /**
   * Get current accuracy for a system
   */
  getSystemAccuracy(system: 'z3' | 'leanaide'): number {
    const key = `${system}_accuracy`;
    return this.accuracyHistory.get(key) || 0.85;
  }
}
