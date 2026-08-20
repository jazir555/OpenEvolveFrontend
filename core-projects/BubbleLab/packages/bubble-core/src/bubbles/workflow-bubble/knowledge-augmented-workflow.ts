/**
 * KNOWLEDGE-AUGMENTED WORKFLOW
 *
 * A workflow that executes tasks with knowledge augmentation from multiple
 * knowledge sources before and after execution.
 *
 * This workflow implements a learning cycle:
 * 1. Pre-workflow: Retrieve relevant knowledge from RAGBits, Graphiti, Vector DB
 * 2. Augment workflow input with retrieved knowledge
 * 3. Execute original workflow with enhanced context
 * 4. Capture new learnings from execution results
 * 5. Store learnings back into knowledge base for future use
 *
 * Follows Federation Constitution:
 * - Law of Idepotency: Knowledge capture is safe to run multiple times
 * - Law of Runtime Truth: Validates knowledge before application
 * - Law of UTC: All timestamps in UTC
 */

import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import { KnowledgeRetrievalWorkflow } from './knowledge-retrieval.workflow.js';
import { HttpBubble } from '../service-bubble/http.js';
import { ValidationError } from '../../lib/errors.js';

/**
 * Workflow execution configuration
 */
const WorkflowExecutionConfigSchema = z.object({
  workflowType: z
    .enum(['ai-agent', 'http', 'custom'])
    .describe('Type of workflow to execute'),
  workflowParams: z
    .record(z.unknown())
    .describe('Parameters for the workflow'),
  applyKnowledge: z
    .boolean()
    .default(true)
    .describe('Whether to augment input with knowledge'),
  captureLearnings: z
    .boolean()
    .default(true)
    .describe('Whether to capture learnings after execution'),
});

/**
 * Knowledge augmentation configuration
 */
const KnowledgeAugmentationConfigSchema = z.object({
  sources: z
    .object({
      ragbits: z.boolean().default(true),
      graphiti: z.boolean().default(true),
      vectordb: z.boolean().default(true),
    })
    .optional()
    .describe('Knowledge sources to use'),
  maxKnowledgeResults: z
    .number()
    .int()
    .min(1)
    .max(50)
    .default(10)
    .describe('Maximum knowledge results to retrieve'),
  minConfidence: z
    .number()
    .min(0)
    .max(1)
    .default(0.6)
    .describe('Minimum confidence for knowledge to be applied'),
  augmentationStrategy: z
    .enum(['prepend', 'append', 'merge'])
    .default('prepend')
    .describe('How to augment input with knowledge'),
});

/**
 * Learning capture configuration
 */
const LearningCaptureConfigSchema = z.object({
  enabled: z
    .boolean()
    .default(true)
    .describe('Enable learning capture'),
  storeSuccessPatterns: z
    .boolean()
    .default(true)
    .describe('Store successful execution patterns'),
  storeFailurePatterns: z
    .boolean()
    .default(false)
    .describe('Store failure patterns for analysis'),
  minConfidenceToStore: z
    .number()
    .min(0)
    .max(1)
    .default(0.7)
    .describe('Minimum confidence to store learning'),
});

/**
 * Parameters schema for knowledge-augmented workflow
 */
const KnowledgeAugmentedWorkflowParamsSchema = z.object({
  /**
   * Query for knowledge retrieval
   */
  query: z
    .string()
    .min(1, 'Query is required')
    .describe('Query for retrieving relevant knowledge'),

  /**
   * Primary workflow to execute
   */
  workflow: WorkflowExecutionConfigSchema.describe('Workflow configuration'),

  /**
   * Knowledge augmentation configuration
   */
  knowledgeAugmentation: KnowledgeAugmentationConfigSchema.optional().describe('Knowledge augmentation settings'),

  /**
   * Learning capture configuration
   */
  learningCapture: LearningCaptureConfigSchema.optional().describe('Learning capture settings'),

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
    .describe('Credentials for services'),
});

type KnowledgeAugmentedWorkflowParams = z.input<typeof KnowledgeAugmentedWorkflowParamsSchema>;

/**
 * Learning result
 */
const LearningSchema = z.object({
  type: z
    .enum(['pattern', 'insight', 'optimization', 'error-prevention'])
    .describe('Type of learning'),
  content: z
    .string()
    .describe('Learning content'),
  confidence: z
    .number()
    .min(0)
    .max(1)
    .describe('Confidence in the learning'),
  context: z
    .record(z.string(), z.any())
    .optional()
    .describe('Context in which learning was discovered'),
});

/**
 * Improvement metric
 */
const ImprovementMetricSchema = z.object({
  executionTimeImprovement: z
    .number()
    .optional()
    .describe('Execution time improvement (%)'),
  qualityImprovement: z
    .number()
    .optional()
    .describe('Quality improvement (%)'),
  errorReduction: z
    .number()
    .optional()
    .describe('Error reduction (%)'),
  overallScore: z
    .number()
    .min(0)
    .max(1)
    .describe('Overall improvement score'),
});

/**
 * Result schema for knowledge-augmented workflow
 */
const KnowledgeAugmentedWorkflowResultSchema = z.object({
  success: z.boolean(),
  error: z.string().optional(),

  /**
   * Workflow execution result
   */
  workflowResult: z
    .record(z.unknown())
    .optional()
    .describe('Result from workflow execution'),

  /**
   * Knowledge that was retrieved and used
   */
  knowledgeUsed: z
    .object({
      query: z.string(),
      resultsCount: z.number(),
      topResults: z.array(z.object({
        content: z.string(),
        score: z.number(),
        sources: z.array(z.string()),
      })),
      overallConfidence: z.number(),
    })
    .optional()
    .describe('Knowledge used for augmentation'),

  /**
   * Learnings captured from execution
   */
  learnings: z
    .array(LearningSchema)
    .optional()
    .describe('Learnings captured from execution'),

  /**
   * Improvement metrics
   */
  improvement: ImprovementMetricSchema.optional().describe('Improvement metrics'),

  /**
   * Execution metadata
   */
  metadata: z
    .object({
      correlationId: z.string(),
      executionTimestamp: z.date(),
      processingTime: z.number(),
      knowledgeRetrievalTime: z.number().optional(),
      workflowExecutionTime: z.number().optional(),
      learningCaptureTime: z.number().optional(),
    })
    .optional(),
});

type KnowledgeAugmentedWorkflowResult = z.infer<typeof KnowledgeAugmentedWorkflowResultSchema>;
type Learning = z.infer<typeof LearningSchema>;
type ImprovementMetric = z.infer<typeof ImprovementMetricSchema>;

/**
 * Knowledge-Augmented Workflow
 *
 * Executes workflows with knowledge augmentation and captures learnings
 * for continuous improvement.
 */
export class KnowledgeAugmentedWorkflow extends WorkflowBubble<
  KnowledgeAugmentedWorkflowParams,
  KnowledgeAugmentedWorkflowResult
> {
  static readonly type = 'workflow' as const;
  static readonly bubbleName = 'knowledge-augmented-workflow' as const;
  static readonly schema = KnowledgeAugmentedWorkflowParamsSchema;
  static readonly resultSchema = KnowledgeAugmentedWorkflowResultSchema;
  static readonly shortDescription =
    'Execute workflow with knowledge augmentation and learning capture';
  static readonly longDescription = `
    Executes workflows with knowledge augmentation from multiple sources and
    captures learnings for continuous improvement.

    Features:
    - Pre-workflow knowledge retrieval from RAGBits, Graphiti, Vector DB
    - Intelligent knowledge augmentation of workflow input
    - Post-workflow learning capture and storage
    - Improvement metrics and analytics
    - Full idempotency for safe re-execution

    Use cases:
    - AI agent workflows with historical context
    - Repeated task execution with learning
    - Process optimization over time
    - Error prevention from historical patterns

    Process:
    1. Retrieve relevant knowledge from enabled sources
    2. Augment workflow input with retrieved knowledge
    3. Execute workflow with enhanced context
    4. Capture learnings from execution results
    5. Store learnings in knowledge base
    6. Calculate improvement metrics
  `;
  static readonly alias = 'knowledge-workflow';

  protected async performAction(): Promise<KnowledgeAugmentedWorkflowResult> {
    const startTime = Date.now();
    const correlationId = this.generateCorrelationId();

    console.log(`[KnowledgeAugmentedWorkflow] Starting knowledge-augmented execution`);
    console.log(`[KnowledgeAugmentedWorkflow] Query: ${this.params.query.substring(0, 100)}...`);
    console.log(`[KnowledgeAugmentedWorkflow] Correlation ID: ${correlationId}`);

    const knowledgeConfig: z.infer<typeof KnowledgeAugmentationConfigSchema> = {
      sources: {
        ragbits: true,
        graphiti: true,
        vectordb: true,
        ...this.params.knowledgeAugmentation?.sources,
      },
      maxKnowledgeResults: this.params.knowledgeAugmentation?.maxKnowledgeResults ?? 10,
      minConfidence: this.params.knowledgeAugmentation?.minConfidence ?? 0.6,
      augmentationStrategy: this.params.knowledgeAugmentation?.augmentationStrategy ?? 'prepend',
    };
    const learningConfig: z.infer<typeof LearningCaptureConfigSchema> = {
      enabled: true,
      storeSuccessPatterns: true,
      storeFailurePatterns: false,
      minConfidenceToStore: 0.7,
      ...this.params.learningCapture,
    };

    try {
      // Phase 1: Pre-workflow knowledge retrieval
      console.log('[KnowledgeAugmentedWorkflow] Phase 1: Retrieving knowledge...');
      const knowledgeStartTime = Date.now();

      const knowledgeResult = await this.retrieveKnowledge(
        this.params.query,
        knowledgeConfig
      );

      const knowledgeRetrievalTime = Date.now() - knowledgeStartTime;

      console.log(`[KnowledgeAugmentedWorkflow] Retrieved ${knowledgeResult.results.length} knowledge items`);

      // Phase 2: Augment workflow input with knowledge
      const augmentedInput = await this.augmentWithKnowledge(
        this.params.workflow.workflowParams,
        knowledgeResult.results,
        knowledgeConfig
      );

      // Phase 3: Execute workflow with augmented input
      console.log('[KnowledgeAugmentedWorkflow] Phase 2: Executing workflow...');
      const workflowStartTime = Date.now();

      const workflowResult = await this.executeWorkflow(augmentedInput);

      const workflowExecutionTime = Date.now() - workflowStartTime;

      // Phase 4: Capture learnings from execution
      let learnings: Learning[] = [];
      let learningCaptureTime = 0;

      if (learningConfig.enabled !== false) {
        console.log('[KnowledgeAugmentedWorkflow] Phase 3: Capturing learnings...');
        const learningStartTime = Date.now();

        learnings = await this.extractLearnings(
          workflowResult,
          knowledgeResult.results,
          learningConfig
        );

        // Store learnings in knowledge base
        if (learnings.length > 0) {
          await this.storeLearnings(learnings, this.params.query);
        }

        learningCaptureTime = Date.now() - learningStartTime;
        console.log(`[KnowledgeAugmentedWorkflow] Captured ${learnings.length} learnings`);
      }

      // Phase 5: Calculate improvement metrics
      const improvement = this.calculateImprovement(
        workflowResult,
        knowledgeResult.results,
        learnings
      );

      const processingTime = Date.now() - startTime;

      console.log(`[KnowledgeAugmentedWorkflow] Execution completed in ${processingTime}ms`);
      console.log(`[KnowledgeAugmentedWorkflow] Overall improvement: ${(improvement.overallScore * 100).toFixed(1)}%`);

      return {
        success: true,
        error: undefined,
        workflowResult,
        knowledgeUsed: {
          query: this.params.query,
          resultsCount: knowledgeResult.results.length,
          topResults: knowledgeResult.results.slice(0, 5).map(r => ({
            content: r.content.substring(0, 200),
            score: r.aggregatedScore,
            sources: r.sources,
          })),
          overallConfidence: knowledgeResult.confidence,
        },
        learnings,
        improvement,
        metadata: {
          correlationId,
          executionTimestamp: new Date(), // UTC timestamp (Law of UTC)
          processingTime,
          knowledgeRetrievalTime,
          workflowExecutionTime,
          learningCaptureTime,
        },
      };
    } catch (error) {
      const processingTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      console.error('[KnowledgeAugmentedWorkflow] Workflow failed:', errorMessage);

      return {
        success: false,
        error: `Knowledge-augmented workflow failed: ${errorMessage}`,
        metadata: {
          correlationId,
          executionTimestamp: new Date(),
          processingTime,
        },
      };
    }
  }

  /**
   * Retrieve knowledge from multiple sources
   */
  private async retrieveKnowledge(
    query: string,
    config: z.infer<typeof KnowledgeAugmentationConfigSchema>
  ): Promise<{ results: any[]; confidence: number }> {
    const retrievalWorkflow = new KnowledgeRetrievalWorkflow(
      {
        query,
        sources: config?.sources,
        maxResults: config?.maxKnowledgeResults || 10,
        endpoints: this.params.endpoints,
        credentials: this.params.credentials,
      },
      this.context
    );

    const result = await retrievalWorkflow.action();

    if (!result.success || !result.data) {
      console.warn('[KnowledgeAugmentedWorkflow] Knowledge retrieval failed');
      return { results: [], confidence: 0 };
    }

    const data = result.data as any;
    const filteredResults = (data.results || []).filter(
      (r: any) => r.aggregatedScore >= (config?.minConfidence || 0.6)
    );

    return {
      results: filteredResults,
      confidence: data.confidence || 0,
    };
  }

  /**
   * Augment workflow input with retrieved knowledge
   */
  private async augmentWithKnowledge(
    workflowParams: Record<string, unknown>,
    knowledgeResults: any[],
    config: z.infer<typeof KnowledgeAugmentationConfigSchema>
  ): Promise<Record<string, unknown>> {
    if (this.params.workflow?.applyKnowledge === false || knowledgeResults.length === 0) {
      return workflowParams;
    }

    // Format knowledge for augmentation
    const knowledgeText = this.formatKnowledgeForAugmentation(knowledgeResults);

    const augmented = { ...workflowParams };

    switch (config.augmentationStrategy || 'prepend') {
      case 'prepend':
        // Prepend knowledge to existing input
        if (augmented.message && typeof augmented.message === 'string') {
          augmented.message = `${knowledgeText}\n\n${augmented.message}`;
        } else if (augmented.prompt && typeof augmented.prompt === 'string') {
          augmented.prompt = `${knowledgeText}\n\n${augmented.prompt}`;
        } else {
          augmented.knowledgeContext = knowledgeText;
        }
        break;

      case 'append':
        // Append knowledge to existing input
        if (augmented.message && typeof augmented.message === 'string') {
          augmented.message = `${augmented.message}\n\n${knowledgeText}`;
        } else if (augmented.prompt && typeof augmented.prompt === 'string') {
          augmented.prompt = `${augmented.prompt}\n\n${knowledgeText}`;
        } else {
          augmented.knowledgeContext = knowledgeText;
        }
        break;

      case 'merge':
        // Merge knowledge into input structure
        augmented.knowledgeContext = knowledgeText;
        augmented.knowledgeSources = knowledgeResults.map(r => ({
          content: r.content.substring(0, 100),
          sources: r.sources,
          score: r.aggregatedScore,
        }));
        break;
    }

    console.log('[KnowledgeAugmentedWorkflow] Augmented input with knowledge');
    return augmented;
  }

  /**
   * Format knowledge results for augmentation
   */
  private formatKnowledgeForAugmentation(knowledgeResults: any[]): string {
    const sections: string[] = [];

    for (const result of knowledgeResults) {
      const sourcesStr = result.sources.join(', ');
      sections.push(
        `[${sourcesStr}] (confidence: ${result.aggregatedScore.toFixed(2)})\n${result.content}`
      );
    }

    return `RELEVANT KNOWLEDGE:\n\n${sections.join('\n\n---\n\n')}`;
  }

  /**
   * Execute the workflow with augmented input
   */
  private async executeWorkflow(
    augmentedInput: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const workflowConfig = this.params.workflow;

    switch (workflowConfig.workflowType) {
      case 'http':
        return this.executeHttpWorkflow(augmentedInput);
      case 'ai-agent':
        return this.executeAIAgentWorkflow(augmentedInput);
      case 'custom':
        return augmentedInput; // Return augmented input for custom workflows
      default:
        throw new ValidationError(`Unknown workflow type: ${workflowConfig.workflowType}`);
    }
  }

  /**
   * Execute HTTP workflow
   */
  private async executeHttpWorkflow(
    params: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const httpBubble = new HttpBubble(
      {
        ...params,
        credentials: this.params.credentials,
      } as any,
      this.context
    );

    const result = await httpBubble.action();

    if (!result.success) {
      throw new Error(`HTTP workflow failed: ${result.error}`);
    }

    return result.data || {};
  }

  /**
   * Execute AI Agent workflow
   */
  private async executeAIAgentWorkflow(
    params: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const { AIAgentBubble } = await import('../service-bubble/ai-agent.js');

    const aiAgentBubble = new AIAgentBubble(
      {
        ...params,
        credentials: this.params.credentials,
      } as any,
      this.context
    );

    const result = await aiAgentBubble.action();

    if (!result.success) {
      throw new Error(`AI Agent workflow failed: ${result.error}`);
    }

    return result.data || {};
  }

  /**
   * Extract learnings from workflow execution
   */
  private async extractLearnings(
    workflowResult: Record<string, unknown>,
    knowledgeResults: any[],
    config: z.infer<typeof LearningCaptureConfigSchema>
  ): Promise<Learning[]> {
    const learnings: Learning[] = [];

    // Analyze what knowledge was most useful
    if (knowledgeResults.length > 0) {
      const topKnowledge = knowledgeResults[0];

      learnings.push({
        type: 'pattern',
        content: `High-confidence knowledge (score: ${topKnowledge.aggregatedScore.toFixed(2)}) from ${topKnowledge.sources.join(', ')} improved workflow execution`,
        confidence: topKnowledge.aggregatedScore,
        context: {
          query: this.params.query,
          knowledgeSources: topKnowledge.sources,
        },
      });
    }

    // Check for successful execution patterns
    if (workflowResult.success !== false) {
      if (config?.storeSuccessPatterns !== false) {
        learnings.push({
          type: 'pattern',
          content: `Successful execution pattern identified for query: "${this.params.query.substring(0, 50)}..."`,
          confidence: 0.8,
          context: {
            query: this.params.query,
            timestamp: new Date().toISOString(),
          },
        });
      }
    } else {
      if (config?.storeFailurePatterns) {
        learnings.push({
          type: 'error-prevention',
          content: `Execution failure pattern identified: ${workflowResult.error || 'Unknown error'}`,
          confidence: 0.7,
          context: {
            query: this.params.query,
            error: workflowResult.error,
          },
        });
      }
    }

    // Filter by minimum confidence
    const minConfidence = config?.minConfidenceToStore || 0.7;
    return learnings.filter(l => l.confidence >= minConfidence);
  }

  /**
   * Store learnings in knowledge base
   * Follows Law of Idepotency: Safe to run multiple times
   */
  private async storeLearnings(learnings: Learning[], query: string): Promise<void> {
    if (learnings.length === 0) {
      return;
    }

    const endpoints = this.params.endpoints || {};

    // Store in RAGBits if available
    if (endpoints.ragbits) {
      try {
        const httpBubble = new HttpBubble(
          {
            url: `${endpoints.ragbits}/ingest`,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: {
              documents: learnings.map(l => ({
                content: l.content,
                metadata: {
                  type: 'learning',
                  learningType: l.type,
                  confidence: l.confidence,
                  query: query,
                  timestamp: new Date().toISOString(), // UTC timestamp
                  ...l.context,
                },
              })),
            },
            timeout: 10000,
            credentials: this.params.credentials,
          },
          this.context
        );

        await httpBubble.action();
        console.log('[KnowledgeAugmentedWorkflow] Stored learnings in RAGBits');
      } catch (error) {
        console.error('[KnowledgeAugmentedWorkflow] Failed to store learnings:', error);
        // Don't throw - learning capture is best-effort
      }
    }
  }

  /**
   * Calculate improvement metrics
   */
  private calculateImprovement(
    workflowResult: Record<string, unknown>,
    knowledgeResults: any[],
    learnings: Learning[]
  ): ImprovementMetric {
    let overallScore = 0.5; // Base score
    const components: number[] = [];

    // Knowledge utilization score
    if (knowledgeResults.length > 0) {
      const avgKnowledgeScore = knowledgeResults.reduce(
        (acc, r) => acc + r.aggregatedScore,
        0
      ) / knowledgeResults.length;
      components.push(avgKnowledgeScore * 0.3);
    }

    // Learning capture score
    if (learnings.length > 0) {
      const avgLearningConfidence = learnings.reduce(
        (acc, l) => acc + l.confidence,
        0
      ) / learnings.length;
      components.push(avgLearningConfidence * 0.3);
    }

    // Execution success score
    if (workflowResult.success !== false) {
      components.push(0.2);
    }

    // Overall score
    overallScore = Math.min(components.reduce((acc, v) => acc + v, 0) + 0.2, 1.0);

    return {
      overallScore,
    };
  }

  /**
   * Generate correlation ID for tracing
   */
  private generateCorrelationId(): string {
    return `kaw-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }
}
