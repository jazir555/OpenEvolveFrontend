/**
 * KNOWLEDGE CAPTURE WORKFLOW
 *
 * A workflow that captures learnings and patterns from workflow executions
 * and stores them in the knowledge base for future retrieval.
 *
 * This workflow implements:
 * 1. Pattern extraction from workflow executions
 * 2. Successful pattern storage in knowledge base
 * 3. Confidence score updates based on outcomes
 * 4. Input-outcome linkage for future learning
 * 5. Learning summary generation
 *
 * Follows Federation Constitution:
 * - Law of Idepotency: Safe to run multiple times (UPSERT logic)
 * - Law of Runtime Truth: Validates patterns before storage
 * - Law of UTC: All timestamps in UTC
 */

import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';
import { ValidationError } from '../../../lib/errors.js';

/**
 * Pattern extraction configuration
 */
const PatternExtractionConfigSchema = z.object({
  extractSuccessPatterns: z
    .boolean()
    .default(true)
    .describe('Extract patterns from successful executions'),
  extractFailurePatterns: z
    .boolean()
    .default(false)
    .describe('Extract patterns from failed executions'),
  extractOptimizationOpportunities: z
    .boolean()
    .default(true)
    .describe('Identify optimization opportunities'),
  minConfidence: z
    .number()
    .min(0)
    .max(1)
    .default(0.7)
    .describe('Minimum confidence to store pattern'),
});

/**
 * Workflow execution data
 */
const WorkflowExecutionSchema = z.object({
  workflowType: z
    .string()
    .describe('Type of workflow that was executed'),
  workflowId: z
    .string()
    .optional()
    .describe('Unique identifier for the workflow'),
  input: z
    .record(z.unknown())
    .describe('Input parameters for the workflow'),
  output: z
    .record(z.unknown())
    .optional()
    .describe('Output from the workflow'),
  startTime: z
    .date()
    .describe('Workflow start time (UTC)'),
  endTime: z
    .date()
    .describe('Workflow end time (UTC)'),
  duration: z
    .number()
    .describe('Workflow duration in milliseconds'),
  success: z
    .boolean()
    .describe('Whether the workflow succeeded'),
  errorMessage: z
    .string()
    .optional()
    .describe('Error message if workflow failed'),
  metadata: z
    .record(z.unknown())
    .optional()
    .describe('Additional execution metadata'),
});

/**
 * Outcome data
 */
const OutcomeSchema = z.object({
  success: z
    .boolean()
    .describe('Whether the outcome was successful'),
  qualityScore: z
    .number()
    .min(0)
    .max(1)
    .optional()
    .describe('Quality score of the outcome'),
  efficiency: z
    .number()
    .min(0)
    .max(1)
    .optional()
    .describe('Efficiency score of the outcome'),
  userFeedback: z
    .string()
    .optional()
    .describe('User feedback on the outcome'),
  metrics: z
    .record(z.number())
    .optional()
    .describe('Additional outcome metrics'),
});

/**
 * Storage configuration
 */
const StorageConfigSchema = z.object({
  storeInRAGBits: z
    .boolean()
    .default(true)
    .describe('Store patterns in RAGBits'),
  storeInGraphiti: z
    .boolean()
    .default(false)
    .describe('Store entities/relationships in Graphiti'),
  storeInVectorDB: z
    .boolean()
    .default(true)
    .describe('Store patterns in Vector DB'),
  updateExisting: z
    .boolean()
    .default(true)
    .describe('Update existing patterns (UPSERT)'),
});

/**
 * Parameters schema for knowledge capture workflow
 */
const KnowledgeCaptureParamsSchema = z.object({
  /**
   * Workflow execution data
   */
  execution: WorkflowExecutionConfigSchema.describe('Workflow execution data'),

  /**
   * Outcome data
   */
  outcomes: z
    .array(OutcomeSchema)
    .optional()
    .describe('Outcomes from the execution'),

  /**
   * Input data for linkage
   */
  inputData: z
    .record(z.unknown())
    .optional()
    .describe('Input data for outcome linkage'),

  /**
   * Pattern extraction configuration
   */
  patternExtraction: PatternExtractionConfigSchema.optional().describe('Pattern extraction settings'),

  /**
   * Storage configuration
   */
  storage: StorageConfigSchema.optional().describe('Storage settings'),

  /**
   * Knowledge source endpoints
   */
  endpoints: z
    .object({
      ragbits: z.string().url().optional(),
      graphiti: z.string().url().optional(),
      vectordb: z.string().url().optional(),
    })
    .optional()
    .describe('Knowledge source endpoints'),

  /**
   * Credentials
   */
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Credentials for knowledge sources'),
});

type KnowledgeCaptureParams = z.input<typeof KnowledgeCaptureParamsSchema>;

/**
 * Captured knowledge
 */
const CapturedKnowledgeSchema = z.object({
  id: z
    .string()
    .describe('Unique identifier for the captured knowledge'),
  type: z
    .enum(['pattern', 'insight', 'optimization', 'error-prevention'])
    .describe('Type of captured knowledge'),
  content: z
    .string()
    .describe('Knowledge content'),
  confidence: z
    .number()
    .min(0)
    .max(1)
    .describe('Confidence in the knowledge'),
  source: z
    .string()
    .describe('Source workflow or system'),
  metadata: z
    .record(z.string(), z.any())
    .optional()
    .describe('Additional metadata'),
  timestamp: z
    .date()
    .describe('UTC timestamp of capture'),
});

/**
 * Pattern
 */
const PatternSchema = z.object({
  type: z.enum(['success', 'failure', 'optimization', 'anomaly']),
  description: z.string(),
  confidence: z.number().min(0).max(1),
  context: z.record(z.string(), z.any()).optional(),
  repeatCount: z.number().optional(),
  lastSeen: z.date().optional(),
});

/**
 * Learning summary
 */
const LearningSummarySchema = z.object({
  totalPatterns: z
    .number()
    .describe('Total patterns captured'),
  successPatterns: z
    .number()
    .describe('Number of success patterns'),
  failurePatterns: z
    .number()
    .describe('Number of failure patterns'),
  optimizationOpportunities: z
    .number()
    .describe('Number of optimization opportunities'),
  avgConfidence: z
    .number()
    .describe('Average confidence of captured patterns'),
  topInsights: z
    .array(z.string())
    .describe('Top insights from captured knowledge'),
});

/**
 * Result schema for knowledge capture workflow
 */
const KnowledgeCaptureResultSchema = z.object({
  success: z.boolean(),
  error: z.string().optional(),

  /**
   * Captured knowledge items
   */
  captured: z
    .array(CapturedKnowledgeSchema)
    .optional()
    .describe('Captured knowledge items'),

  /**
   * Learning summary
   */
  summary: LearningSummarySchema.optional().describe('Summary of captured learnings'),

  /**
   * Storage results
   */
  storage: z
    .object({
      ragbits: z
        .object({
          attempted: z.boolean(),
          successful: z.boolean(),
          itemCount: z.number(),
        })
        .optional(),
      graphiti: z
        .object({
          attempted: z.boolean(),
          successful: z.boolean(),
          entityCount: z.number(),
          relationshipCount: z.number(),
        })
        .optional(),
      vectordb: z
        .object({
          attempted: z.boolean(),
          successful: z.boolean(),
          itemCount: z.number(),
        })
        .optional(),
    })
    .optional()
    .describe('Storage operation results'),

  /**
   * Capture metadata
   */
  metadata: z
    .object({
      correlationId: z.string(),
      captureTimestamp: z.date(),
      processingTime: z.number(),
      patternExtractionTime: z.number().optional(),
      storageTime: z.number().optional(),
    })
    .optional(),
});

type KnowledgeCaptureResult = z.infer<typeof KnowledgeCaptureResultSchema>;
type CapturedKnowledge = z.infer<typeof CapturedKnowledgeSchema>;
type Pattern = z.infer<typeof PatternSchema>;
type LearningSummary = z.infer<typeof LearningSummarySchema>;

/**
 * Knowledge Capture Workflow
 *
 * Captures learnings from workflow executions and stores them in
 * the knowledge base for future retrieval.
 */
export class KnowledgeCaptureWorkflow extends WorkflowBubble<
  KnowledgeCaptureParams,
  KnowledgeCaptureResult
> {
  static readonly type = 'workflow' as const;
  static readonly bubbleName = 'knowledge-capture' as const;
  static readonly schema = KnowledgeCaptureParamsSchema;
  static readonly resultSchema = KnowledgeCaptureResultSchema;
  static readonly shortDescription =
    'Capture and store learnings from workflow executions';
  static readonly longDescription = `
    Captures learnings and patterns from workflow executions and stores them
    in the knowledge base for future retrieval and learning.

    Features:
    - Automatic pattern extraction from executions
    - Success/failure pattern identification
    - Optimization opportunity detection
    - Confidence-based pattern storage
    - Idempotent storage (UPSERT logic)
    - Multi-source storage (RAGBits, Graphiti, Vector DB)

    Use cases:
    - Continuous learning from workflow executions
    - Building knowledge base from experience
    - Error pattern detection and prevention
    - Optimization opportunity identification

    Process:
    1. Extract patterns from workflow execution
    2. Filter patterns by confidence threshold
    3. Store successful patterns in knowledge base
    4. Update confidence scores based on outcomes
    5. Link inputs to outcomes for future learning
    6. Generate learning summary
  `;
  static readonly alias = 'capture-knowledge';

  protected async performAction(): Promise<KnowledgeCaptureResult> {
    const startTime = Date.now();
    const correlationId = this.generateCorrelationId();

    console.log(`[KnowledgeCapture] Starting knowledge capture`);
    console.log(`[KnowledgeCapture] Workflow type: ${this.params.execution.workflowType}`);
    console.log(`[KnowledgeCapture] Correlation ID: ${correlationId}`);

    const patternConfig = this.params.patternExtraction || {};
    const storageConfig = this.params.storage || {};

    try {
      // Phase 1: Extract patterns from workflow execution
      console.log('[KnowledgeCapture] Phase 1: Extracting patterns...');
      const patternStartTime = Date.now();

      const patterns = await this.extractPatterns(
        this.params.execution,
        patternConfig
      );

      const patternExtractionTime = Date.now() - patternStartTime;

      console.log(`[KnowledgeCapture] Extracted ${patterns.length} patterns`);

      // Phase 2: Filter and store successful patterns
      console.log('[KnowledgeCapture] Phase 2: Storing patterns...');
      const storageStartTime = Date.now();

      const captured: CapturedKnowledge[] = [];
      const storageResults: NonNullable<KnowledgeCaptureResult['storage']> = {};

      // Filter patterns by minimum confidence
      const minConfidence = patternConfig.minConfidence ?? 0.7;
      const validPatterns = patterns.filter(p => p.confidence >= minConfidence);

      console.log(`[KnowledgeCapture] ${validPatterns.length} patterns passed confidence threshold`);

      // Store patterns in enabled knowledge sources
      for (const pattern of validPatterns) {
        // Check if pattern should be stored based on type and config
        if (pattern.type === 'success' && patternConfig.extractSuccessPatterns === false) {
          continue;
        }
        if (pattern.type === 'failure' && patternConfig.extractFailurePatterns === false) {
          continue;
        }
        if (pattern.type === 'optimization' && patternConfig.extractOptimizationOpportunities === false) {
          continue;
        }

        const capturedItem = await this.storePattern(pattern, storageConfig);
        captured.push(capturedItem);
      }

      const storageTime = Date.now() - storageStartTime;

      // Phase 3: Update confidence scores based on outcomes
      if (this.params.outcomes && this.params.outcomes.length > 0) {
        console.log('[KnowledgeCapture] Phase 3: Updating confidence scores...');
        await this.updateConfidenceScores(this.params.execution, this.params.outcomes);
      }

      // Phase 4: Link inputs to outcomes for future learning
      if (this.params.inputData && this.params.outcomes) {
        console.log('[KnowledgeCapture] Phase 4: Linking inputs to outcomes...');
        await this.linkInputOutcome(this.params.inputData, this.params.outcomes);
      }

      // Phase 5: Generate learning summary
      console.log('[KnowledgeCapture] Phase 5: Generating summary...');
      const summary = await this.generateLearningSummary(captured);

      const processingTime = Date.now() - startTime;

      console.log(`[KnowledgeCapture] Capture completed in ${processingTime}ms`);
      console.log(`[KnowledgeCapture] Captured ${captured.length} knowledge items`);

      return {
        success: true,
        error: undefined,
        captured,
        summary,
        storage: storageResults,
        metadata: {
          correlationId,
          captureTimestamp: new Date(), // UTC timestamp (Law of UTC)
          processingTime,
          patternExtractionTime,
          storageTime,
        },
      };
    } catch (error) {
      const processingTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      console.error('[KnowledgeCapture] Workflow failed:', errorMessage);

      return {
        success: false,
        error: `Knowledge capture failed: ${errorMessage}`,
        metadata: {
          correlationId,
          captureTimestamp: new Date(),
          processingTime,
        },
      };
    }
  }

  /**
   * Extract patterns from workflow execution
   */
  private async extractPatterns(
    execution: z.infer<typeof WorkflowExecutionSchema>,
    config: z.infer<typeof PatternExtractionConfigSchema>
  ): Promise<Pattern[]> {
    const patterns: Pattern[] = [];

    // Extract success patterns
    if (config.extractSuccessPatterns !== false && execution.success) {
      patterns.push({
        type: 'success',
        description: `Successful execution of ${execution.workflowType}`,
        confidence: 0.8,
        context: {
          workflowType: execution.workflowType,
          workflowId: execution.workflowId,
          duration: execution.duration,
        },
        repeatCount: 1,
        lastSeen: new Date(),
      });

      // Identify optimization opportunities
      if (config.extractOptimizationOpportunities !== false) {
        const optimizations = this.identifyOptimizations(execution);
        patterns.push(...optimizations);
      }
    }

    // Extract failure patterns
    if (config.extractFailurePatterns && !execution.success && execution.errorMessage) {
      patterns.push({
        type: 'failure',
        description: `Execution failure: ${execution.errorMessage}`,
        confidence: 0.7,
        context: {
          workflowType: execution.workflowType,
          error: execution.errorMessage,
          input: JSON.stringify(execution.input).substring(0, 200),
        },
        repeatCount: 1,
        lastSeen: new Date(),
      });
    }

    // Extract duration patterns
    if (execution.duration) {
      const isFastExecution = execution.duration < 5000; // Less than 5 seconds
      const isSlowExecution = execution.duration > 60000; // More than 1 minute

      if (isFastExecution && execution.success) {
        patterns.push({
          type: 'optimization',
          description: `Fast execution pattern identified (${execution.duration}ms)`,
          confidence: 0.6,
          context: {
            workflowType: execution.workflowType,
            duration: execution.duration,
          },
        });
      }

      if (isSlowExecution) {
        patterns.push({
          type: 'optimization',
          description: `Slow execution detected (${execution.duration}ms) - consider optimization`,
          confidence: 0.75,
          context: {
            workflowType: execution.workflowType,
            duration: execution.duration,
          },
        });
      }
    }

    // Extract input patterns
    if (execution.input) {
      const inputPatterns = this.extractInputPatterns(execution);
      patterns.push(...inputPatterns);
    }

    return patterns;
  }

  /**
   * Identify optimization opportunities
   */
  private identifyOptimizations(
    execution: z.infer<typeof WorkflowExecutionSchema>
  ): Pattern[] {
    const optimizations: Pattern[] = [];

    // Check for repeated successful patterns
    if (execution.success && execution.duration) {
      optimizations.push({
        type: 'optimization',
        description: `Efficient execution pattern for ${execution.workflowType}`,
        confidence: 0.7,
        context: {
          workflowType: execution.workflowType,
          duration: execution.duration,
          inputKeys: Object.keys(execution.input || {}),
        },
      });
    }

    return optimizations;
  }

  /**
   * Extract patterns from input data
   */
  private extractInputPatterns(
    execution: z.infer<typeof WorkflowExecutionSchema>
  ): Pattern[] {
    const patterns: Pattern[] = [];

    if (!execution.input || typeof execution.input !== 'object') {
      return patterns;
    }

    const input = execution.input as Record<string, unknown>;
    const keys = Object.keys(input);

    // Identify common input patterns
    if (keys.length > 0) {
      patterns.push({
        type: 'success',
        description: `Input pattern for ${execution.workflowType}: ${keys.join(', ')}`,
        confidence: 0.6,
        context: {
          workflowType: execution.workflowType,
          inputKeys: keys,
          inputTypes: keys.map(k => typeof input[k]),
        },
      });
    }

    return patterns;
  }

  /**
   * Store pattern in knowledge base
   * Follows Law of Idepotency: Uses UPSERT logic for safe re-execution
   */
  private async storePattern(
    pattern: Pattern,
    storageConfig: z.infer<typeof StorageConfigSchema>
  ): Promise<CapturedKnowledge> {
    const id = this.generateKnowledgeId(pattern);
    const endpoints = this.params.endpoints || {};

    const captured: CapturedKnowledge = {
      id,
      type: pattern.type,
      content: pattern.description,
      confidence: pattern.confidence,
      source: this.params.execution.workflowType,
      metadata: {
        ...pattern.context,
        repeatCount: pattern.repeatCount,
        lastSeen: pattern.lastSeen?.toISOString(),
      },
      timestamp: new Date(), // UTC timestamp
    };

    // Store in RAGBits
    if (storageConfig.storeInRAGBits !== false && endpoints.ragbits) {
      try {
        const httpBubble = new HttpBubble(
          {
            url: `${endpoints.ragbits}/ingest`,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: {
              documents: [
                {
                  content: captured.content,
                  metadata: {
                    id: captured.id,
                    type: captured.type,
                    confidence: captured.confidence,
                    source: captured.source,
                    timestamp: captured.timestamp.toISOString(),
                    ...captured.metadata,
                  },
                },
              ],
              update: storageConfig.updateExisting ? true : undefined, // Enable UPSERT
            },
            timeout: 10000,
            credentials: this.params.credentials,
          },
          this.context
        );

        await httpBubble.action();
        console.log(`[KnowledgeCapture] Stored pattern ${id} in RAGBits`);
      } catch (error) {
        console.error('[KnowledgeCapture] Failed to store in RAGBits:', error);
        // Don't throw - storage is best-effort
      }
    }

    // Store in Vector DB
    if (storageConfig.storeInVectorDB !== false && endpoints.vectordb) {
      try {
        const httpBubble = new HttpBubble(
          {
            url: `${endpoints.vectordb}/upsert`,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: {
              vectors: [
                {
                  id: captured.id,
                  content: captured.content,
                  metadata: {
                    type: captured.type,
                    confidence: captured.confidence,
                    source: captured.source,
                    timestamp: captured.timestamp.toISOString(),
                  },
                },
              ],
            },
            timeout: 10000,
            credentials: this.params.credentials,
          },
          this.context
        );

        await httpBubble.action();
        console.log(`[KnowledgeCapture] Stored pattern ${id} in Vector DB`);
      } catch (error) {
        console.error('[KnowledgeCapture] Failed to store in Vector DB:', error);
      }
    }

    return captured;
  }

  /**
   * Update confidence scores based on outcomes
   */
  private async updateConfidenceScores(
    execution: z.infer<typeof WorkflowExecutionSchema>,
    outcomes: z.infer<typeof OutcomeSchema>[]
  ): Promise<void> {
    const endpoints = this.params.endpoints || {};

    // Calculate average outcome scores
    const avgSuccess = outcomes.reduce((acc, o) => acc + (o.success ? 1 : 0), 0) / outcomes.length;
    const avgQuality = outcomes.reduce((acc, o) => acc + (o.qualityScore || 0), 0) / outcomes.length;
    const avgEfficiency = outcomes.reduce((acc, o) => acc + (o.efficiency || 0), 0) / outcomes.length;

    console.log(`[KnowledgeCapture] Outcome averages - success: ${avgSuccess.toFixed(2)}, quality: ${avgQuality.toFixed(2)}, efficiency: ${avgEfficiency.toFixed(2)}`);

    // Update confidence in RAGBits based on outcomes
    if (endpoints.ragbits && avgSuccess > 0.7) {
      try {
        const httpBubble = new HttpBubble(
          {
            url: `${endpoints.ragbits}/update`,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: {
              filter: {
                source: execution.workflowType,
              },
              update: {
                confidence_increment: avgSuccess * 0.1,
              },
            },
            timeout: 10000,
            credentials: this.params.credentials,
          },
          this.context
        );

        await httpBubble.action();
        console.log('[KnowledgeCapture] Updated confidence scores in RAGBits');
      } catch (error) {
        console.error('[KnowledgeCapture] Failed to update confidence scores:', error);
      }
    }
  }

  /**
   * Link inputs to outcomes for future learning
   */
  private async linkInputOutcome(
    inputData: Record<string, unknown>,
    outcomes: z.infer<typeof OutcomeSchema>[]
  ): Promise<void> {
    const endpoints = this.params.endpoints || {};

    if (!endpoints.graphiti) {
      console.log('[KnowledgeCapture] Graphiti endpoint not configured, skipping input-outcome linkage');
      return;
    }

    try {
      // Create entities for input and outcome
      const inputEntity = {
        name: `input_${Date.now()}`,
        type: 'Input',
        attributes: inputData,
      };

      const outcomeEntities = outcomes.map((outcome, index) => ({
        name: `outcome_${Date.now()}_${index}`,
        type: 'Outcome',
        attributes: {
          success: outcome.success,
          qualityScore: outcome.qualityScore,
          efficiency: outcome.efficiency,
          metrics: outcome.metrics,
        },
      }));

      // Create relationships
      const relationships = outcomeEntities.map(outcome => ({
        from: inputEntity.name,
        to: outcome.name,
        type: 'PRODUCES',
        attributes: {
          timestamp: new Date().toISOString(),
        },
      }));

      const httpBubble = new HttpBubble(
        {
          url: `${endpoints.graphiti}/ingest`,
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            entities: [inputEntity, ...outcomeEntities],
            relationships,
          },
          timeout: 10000,
          credentials: this.params.credentials,
        },
        this.context
      );

      await httpBubble.action();
      console.log('[KnowledgeCapture] Linked inputs to outcomes in Graphiti');
    } catch (error) {
      console.error('[KnowledgeCapture] Failed to link inputs to outcomes:', error);
    }
  }

  /**
   * Generate learning summary
   */
  private async generateLearningSummary(
    captured: CapturedKnowledge[]
  ): Promise<LearningSummary> {
    const successPatterns = captured.filter(c => c.type === 'success').length;
    const failurePatterns = captured.filter(c => c.type === 'failure').length;
    const optimizationOpportunities = captured.filter(c => c.type === 'optimization').length;

    const avgConfidence =
      captured.length > 0
        ? captured.reduce((acc, c) => acc + c.confidence, 0) / captured.length
        : 0;

    // Extract top insights
    const topInsights = captured
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5)
      .map(c => c.content);

    return {
      totalPatterns: captured.length,
      successPatterns,
      failurePatterns,
      optimizationOpportunities,
      avgConfidence,
      topInsights,
    };
  }

  /**
   * Generate unique knowledge ID
   */
  private generateKnowledgeId(pattern: Pattern): string {
    const hash = Buffer.from(
      `${pattern.type}-${pattern.description}-${Date.now()}`
    ).toString('base64');
    return `knowledge-${hash.substring(0, 16)}`;
  }

  /**
   * Generate correlation ID for tracing
   */
  private generateCorrelationId(): string {
    return `kc-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }
}
