/**
 * Knowledge Workflow Bubbles Index
 *
 * Exports all knowledge-related workflow bubbles for easy importing.
 *
 * Knowledge Flow:
 * 1. KnowledgeRetrievalWorkflow - Retrieve relevant knowledge from multiple sources
 * 2. KnowledgeAugmentedWorkflow - Execute workflows with knowledge augmentation
 * 3. KnowledgeCaptureWorkflow - Capture and store learnings from executions
 *
 * These workflows can be composed to create learning cycles:
 * - Retrieve knowledge → Augment workflow → Capture learnings → Store in knowledge base
 * - Future workflows benefit from accumulated knowledge
 */

export { KnowledgeRetrievalWorkflow } from './knowledge-retrieval.workflow.js';
export { KnowledgeAugmentedWorkflow } from './knowledge-augmented-workflow.js';
export { KnowledgeCaptureWorkflow } from './knowledge-capture.workflow.js';

/**
 * Knowledge-aware pipeline workflow composition
 *
 * Combines knowledge retrieval, augmented execution, and learning capture
 * into a single learning cycle.
 */
export const KnowledgeAwarePipeline = {
  name: 'knowledge-aware-pipeline',
  description: 'Complete learning cycle with knowledge retrieval, augmented execution, and learning capture',

  steps: [
    {
      bubble: 'knowledge-retrieval' as const,
      description: 'Retrieve relevant knowledge from RAGBits, Graphiti, and Vector DB',
      config: {
        sources: {
          ragbits: true,
          graphiti: true,
          vectordb: true,
        },
        maxResults: 10,
      },
    },
    {
      bubble: 'knowledge-augmented-workflow' as const,
      description: 'Execute workflow with knowledge augmentation',
      config: {
        applyKnowledge: true,
        captureLearnings: true,
      },
    },
    {
      bubble: 'knowledge-capture' as const,
      description: 'Capture and store learnings from execution',
      config: {
        storeSuccessPatterns: true,
        storeInRAGBits: true,
        storeInVectorDB: true,
      },
    },
  ],
};

/**
 * Multi-stage knowledge augmentation workflow
 *
 * Applies knowledge at different stages of workflow execution
 */
export const MultiStageKnowledgeWorkflow = {
  name: 'multi-stage-knowledge',
  description: 'Apply knowledge at multiple stages of workflow execution',

  stages: [
    {
      name: 'pre-processing',
      description: 'Retrieve domain best practices before execution',
      knowledgeQuery: 'domain best practices',
      applyTo: 'input-preparation',
      config: {
        sources: {
          ragbits: true,
          graphiti: false,
          vectordb: false,
        },
      },
    },
    {
      name: 'execution',
      description: 'Find similar historical executions during processing',
      knowledgeQuery: 'similar historical executions',
      applyTo: 'workflow-execution',
      config: {
        sources: {
          ragbits: false,
          graphiti: false,
          vectordb: true,
        },
      },
    },
    {
      name: 'post-processing',
      description: 'Apply validation patterns after execution',
      knowledgeQuery: 'validation patterns',
      applyTo: 'result-validation',
      config: {
        sources: {
          ragbits: true,
          graphiti: true,
          vectordb: false,
        },
      },
    },
  ],
};

/**
 * Adaptive knowledge workflow
 *
 * Dynamically adapts based on retrieved knowledge and execution feedback
 */
export const AdaptiveKnowledgeWorkflow = {
  name: 'adaptive-knowledge',
  description: 'Workflow that adapts based on knowledge and feedback',

  steps: [
    {
      bubble: 'knowledge-retrieval' as const,
      description: 'Retrieve relevant knowledge with adaptive strategy',
      config: {
        strategy: 'adaptive',
        sources: {
          ragbits: true,
          graphiti: true,
          vectordb: true,
        },
      },
    },
    {
      bubble: 'knowledge-augmented-workflow' as const,
      description: 'Execute with adaptation enabled',
      config: {
        adaptationEnabled: true,
        feedbackLoop: true,
        applyKnowledge: true,
        captureLearnings: true,
      },
    },
    {
      bubble: 'knowledge-capture' as const,
      description: 'Store learnings and update strategies',
      config: {
        updateStrategy: true,
        learnFromFeedback: true,
        storeSuccessPatterns: true,
        storeInRAGBits: true,
        storeInVectorDB: true,
      },
    },
  ],
};

/**
 * Continuous learning workflow
 *
 * Iterative workflow that improves over time through learning
 */
export const ContinuousLearningWorkflow = {
  name: 'continuous-learning',
  description: 'Workflow that continuously learns from executions',

  cycles: [
    {
      name: 'learning-cycle-1',
      description: 'First learning iteration',
      query: 'Initial execution context',
      execute: true,
      capture: true,
      improve: true,
    },
  ],

  improvementStrategy: {
    minImprovementThreshold: 0.1, // 10% improvement
    maxIterations: 10,
    convergenceCriteria: 'confidence',
  },
};
