/**
 * Orchestration Layer Integration Examples
 *
 * Complete examples showing how to integrate the orchestration layer
 * with various adapters and workflows.
 */

import {
  eventBus,
  workflowEngine,
  PREDEFINED_WORKFLOWS,
  deadLetterQueue,
  correlationTracker
} from './index';
import { createBaseEvent, Event } from './event-types';
import { Logger } from '../lib/logger';

const logger = new Logger('orchestration-example');

// ============================================================================
// Example 1: Basic Event Publishing and Subscription
// ============================================================================

async function example1_BasicEvents() {
  logger.info('Example 1: Basic Event Publishing and Subscription');

  // Subscribe to KnowledgeExtracted events
  const subscription = eventBus.subscribe('KnowledgeExtracted', async (event) => {
    if (event.type === 'KnowledgeExtracted') {
      logger.info('Knowledge chunks extracted', {
        correlation_id: event.correlation_id,
        document_id: event.data.document_id,
        chunk_count: event.data.chunk_count
      });

      // Process the chunks
      for (const chunk of event.data.chunks) {
        logger.debug('Processing chunk', {
          chunk_id: chunk.chunk_id,
          content_length: chunk.content.length
        });
      }
    }
  });

  // Create and publish an event
  const correlationId = correlationTracker.generateCorrelationId();

  const event = createBaseEvent(
    'KnowledgeExtracted',
    'ragbits-adapter',
    correlationId,
    {
      document_id: 'doc-123',
      chunk_count: 5,
      chunks: [
        {
          chunk_id: 'chunk-1',
          content: 'This is the first chunk of text.',
          metadata: { index: 0, page: 1 }
        },
        {
          chunk_id: 'chunk-2',
          content: 'This is the second chunk of text.',
          metadata: { index: 1, page: 1 }
        }
      ],
      extraction_method: 'recursive'
    }
  );

  await eventBus.publish(event);

  // Cleanup
  eventBus.unsubscribe(subscription.subscriptionId);
}

// ============================================================================
// Example 2: RAG Pipeline with Event Chain
// ============================================================================

async function example2_RAGPipeline() {
  logger.info('Example 2: RAG Pipeline with Event Chain');

  const correlationContext = correlationTracker.createContext({
    workflow: 'rag-pipeline',
    document_id: 'doc-456'
  });

  // Step 1: Subscribe to KnowledgeExtracted → Trigger Vector Indexing
  eventBus.subscribe('KnowledgeExtracted', async (event) => {
    if (event.type === 'KnowledgeExtracted') {
      logger.info('Triggering vector indexing', {
        correlation_id: event.correlation_id,
        chunk_count: event.data.chunk_count
      });

      // Simulate creating embeddings
      const embeddings = event.data.chunks.map(chunk => ({
        chunk_id: chunk.chunk_id,
        embedding: Array(1536).fill(0).map(() => Math.random()), // Mock embedding
        metadata: chunk.metadata
      }));

      // Publish VectorIndexed event
      const vectorEvent = createBaseEvent(
        'VectorIndexed',
        'vector-db-adapter',
        event.correlation_id,
        {
          vector_db_type: 'chroma',
          index_id: 'idx-123',
          embedding_count: embeddings.length,
          embedding_model: 'text-embedding-ada-002',
          dimension: 1536,
          index_type: 'create'
        }
      );

      await eventBus.publish(vectorEvent);
    }
  });

  // Step 2: Subscribe to VectorIndexed → Trigger Graph Update
  eventBus.subscribe('VectorIndexed', async (event) => {
    if (event.type === 'VectorIndexed') {
      logger.info('Triggering knowledge graph update', {
        correlation_id: event.correlation_id,
        embedding_count: event.data.embedding_count
      });

      // Publish GraphUpdated event
      const graphEvent = createBaseEvent(
        'GraphUpdated',
        'graphiti-adapter',
        event.correlation_id,
        {
          graph_id: 'graph-789',
          update_type: 'node_added',
          node_count: event.data.embedding_count,
          edge_count: 0,
          graph_system: 'graphiti',
          changes: event.data.embedding_count === 0 ? [] : [
            {
              type: 'node',
              action: 'added',
              id: 'node-1'
            }
          ]
        }
      );

      await eventBus.publish(graphEvent);
    }
  });

  // Step 3: Start the pipeline by publishing KnowledgeExtracted event
  const startEvent = createBaseEvent(
    'KnowledgeExtracted',
    'ragbits-adapter',
    correlationContext.correlation_id,
    {
      document_id: 'doc-456',
      chunk_count: 3,
      chunks: [
        {
          chunk_id: 'chunk-1',
          content: 'Machine learning is a subset of AI.',
          metadata: {}
        },
        {
          chunk_id: 'chunk-2',
          content: 'Neural networks are inspired by biological neurons.',
          metadata: {}
        },
        {
          chunk_id: 'chunk-3',
          content: 'Deep learning uses multiple layers of neural networks.',
          metadata: {}
        }
      ],
      extraction_method: 'semantic'
    }
  );

  await eventBus.publish(startEvent);

  logger.info('RAG pipeline started', {
    correlation_id: correlationContext.correlation_id
  });
}

// ============================================================================
// Example 3: Workflow Execution
// ============================================================================

async function example3_WorkflowExecution() {
  logger.info('Example 3: Workflow Execution');

  // Execute predefined RAG pipeline workflow
  const result = await workflowEngine.execute(
    PREDEFINED_WORKFLOWS['rag-pipeline'],
    {
      document_id: 'doc-789',
      document_path: '/data/documents/doc-789.pdf',
      metadata: {
        title: 'Introduction to Machine Learning',
        author: 'OpenEvolve',
        date: '2025-01-15'
      }
    },
    correlationTracker.createContext({
      workflow: 'rag-pipeline',
      triggered_by: 'user-api'
    })
  );

  logger.info('Workflow execution completed', {
    execution_id: result.execution_id,
    state: result.state,
    duration_ms: result.duration_ms,
    steps_completed: result.steps_completed,
    steps_failed: result.steps_failed
  });

  if (result.state === 'completed') {
    logger.info('✓ Workflow completed successfully', {
      output: result.output_data
    });
  } else if (result.state === 'failed') {
    logger.error('✗ Workflow failed', undefined, {
      error: result.error,
      execution_id: result.execution_id
    });
  }
}

// ============================================================================
// Example 4: Custom Workflow Definition
// ============================================================================

async function example4_CustomWorkflow() {
  logger.info('Example 4: Custom Workflow Definition');

  // Define custom workflow for document analysis
  const analysisWorkflow = {
    workflow_id: 'document-analysis',
    workflow_name: 'Document Analysis',
    description: 'Analyze document and extract insights',
    steps: [
      {
        step_id: 'extract-text',
        step_name: 'Extract Text Content',
        service: 'ragbits-adapter',
        operation: 'extract-text',
        handler: async (context: any) => {
          logger.info('Extracting text from document', {
            document_id: context.input_data.document_id
          });

          // Simulate text extraction
          await new Promise(resolve => setTimeout(resolve, 100));

          return {
            text: 'This is the extracted text content.',
            length: 100,
            language: 'en'
          };
        },
        timeout_ms: 30000,
        retry_on_failure: true,
        max_retries: 2
      },
      {
        step_id: 'analyze-sentiment',
        step_name: 'Analyze Sentiment',
        service: 'analysis-service',
        operation: 'sentiment-analysis',
        handler: async (context: any) => {
          const textData = context.step_results.get('extract-text');

          logger.info('Analyzing sentiment', {
            text_length: textData.length
          });

          // Simulate sentiment analysis
          await new Promise(resolve => setTimeout(resolve, 50));

          return {
            sentiment: 'positive',
            confidence: 0.85,
            emotions: ['joy', 'optimism']
          };
        },
        timeout_ms: 15000,
        retry_on_failure: true
      },
      {
        step_id: 'extract-keywords',
        step_name: 'Extract Keywords',
        service: 'nlp-service',
        operation: 'keyword-extraction',
        handler: async (context: any) => {
          const textData = context.step_results.get('extract-text');

          logger.info('Extracting keywords', {
            text_length: textData.length
          });

          // Simulate keyword extraction
          await new Promise(resolve => setTimeout(resolve, 50));

          return {
            keywords: ['machine learning', 'AI', 'neural networks'],
            scores: [0.95, 0.88, 0.82]
          };
        },
        timeout_ms: 15000,
        retry_on_failure: true
      }
    ],
    parallel: false,
    on_failure: 'stop' as const,
    timeout_ms: 60000
  };

  // Execute custom workflow
  const result = await workflowEngine.execute(
    analysisWorkflow,
    {
      document_id: 'doc-analysis-1',
      document_path: '/data/docs/sample.pdf'
    }
  );

  logger.info('Custom workflow completed', {
    state: result.state,
    output: result.output_data
  });
}

// ============================================================================
// Example 5: Error Handling with DLQ
// ============================================================================

async function example5_ErrorHandling() {
  logger.info('Example 5: Error Handling with DLQ');

  // Simulate event processing with error handling
  async function processEventWithDLQ(event: Event) {
    try {
      logger.info('Processing event', {
        event_id: event.id,
        event_type: event.type
      });

      // Simulate processing that might fail
      if (Math.random() > 0.5) {
        throw new Error('Random processing failure');
      }

      logger.info('Event processed successfully', {
        event_id: event.id
      });
    } catch (error) {
      logger.error('Event processing failed', error as Error, {
        event_id: event.id,
        event_type: event.type
      });

      // Send to DLQ
      await deadLetterQueue.enqueue(event, error as Error, {
        handler: 'example-handler',
        operation: 'process-event'
      });
    }
  }

  // Create and process events
  const events = [
    createBaseEvent('KnowledgeExtracted', 'test', 'corr-1', { document_id: '1' }),
    createBaseEvent('VectorIndexed', 'test', 'corr-2', { index_id: '2' }),
    createBaseEvent('GraphUpdated', 'test', 'corr-3', { graph_id: '3' })
  ];

  for (const event of events) {
    await processEventWithDLQ(event);
  }

  // Check DLQ stats
  const stats = deadLetterQueue.getStats();
  logger.info('DLQ Statistics', stats);

  // Process DLQ retries
  const retried = await deadLetterQueue.processRetry(async (event) => {
    logger.info('Retrying event from DLQ', {
      event_id: event.id,
      event_type: event.type
    });

    // Retry processing (will succeed this time)
    logger.info('DLQ event processed successfully', {
      event_id: event.id
    });
  });

  logger.info('DLQ retry completed', {
    events_retried: retried
  });
}

// ============================================================================
// Example 6: Correlation Tracking
// ============================================================================

async function example6_CorrelationTracking() {
  logger.info('Example 6: Correlation Tracking');

  // Create correlation context
  const context = correlationTracker.createContext({
    user_id: 'user-123',
    workflow: 'multi-step-pipeline',
    trigger: 'api-request'
  });

  logger.info('Created correlation context', {
    correlation_id: context.correlation_id,
    trace_id: context.trace_id
  });

  // Record service calls
  correlationTracker.recordServiceCall(context, 'ragbits-adapter', 'extract-chunks');
  correlationTracker.recordServiceCall(context, 'vector-db-adapter', 'create-embeddings');
  correlationTracker.recordServiceCall(context, 'vector-db-adapter', 'index-embeddings');
  correlationTracker.recordServiceCall(context, 'graphiti-adapter', 'update-graph');

  // Create distributed trace spans
  const span1 = correlationTracker.createSpan(
    context.trace_id!,
    undefined,
    'orchestration',
    'process-document',
    { document_id: 'doc-123' }
  );

  const span2 = correlationTracker.createSpan(
    context.trace_id!,
    span1.span_id,
    'vector-db-adapter',
    'index-embeddings',
    { index_id: 'idx-123' }
  );

  // Complete spans
  correlationTracker.completeSpan(span1, 'ok');
  correlationTracker.completeSpan(span2, 'ok');

  // Get trace
  const trace = correlationTracker.getTrace(context.trace_id!);
  logger.info('Distributed trace', {
    trace_id: context.trace_id,
    span_count: trace?.length
  });

  // Calculate duration
  const duration = correlationTracker.calculateDuration(context);
  logger.info('Request duration', {
    correlation_id: context.correlation_id,
    duration_ms: duration,
    service_path_count: context.service_path.length
  });

  // Format for logging
  const logContext = correlationTracker.formatForLogging(context);
  logger.info('Request completed', logContext);
}

// ============================================================================
// Example 7: Parallel Workflow Execution
// ============================================================================

async function example7_ParallelWorkflow() {
  logger.info('Example 7: Parallel Workflow Execution');

  // Define parallel workflow
  const parallelWorkflow = {
    workflow_id: 'parallel-processing',
    workflow_name: 'Parallel Processing',
    description: 'Execute multiple independent tasks in parallel',
    steps: [
      {
        step_id: 'task-a',
        step_name: 'Task A',
        service: 'service-a',
        operation: 'process',
        handler: async (context: any) => {
          logger.info('Executing Task A');
          await new Promise(resolve => setTimeout(resolve, 100));
          return { task: 'A', result: 'completed' };
        },
        timeout_ms: 30000
      },
      {
        step_id: 'task-b',
        step_name: 'Task B',
        service: 'service-b',
        operation: 'process',
        handler: async (context: any) => {
          logger.info('Executing Task B');
          await new Promise(resolve => setTimeout(resolve, 150));
          return { task: 'B', result: 'completed' };
        },
        timeout_ms: 30000
      },
      {
        step_id: 'task-c',
        step_name: 'Task C',
        service: 'service-c',
        operation: 'process',
        handler: async (context: any) => {
          logger.info('Executing Task C');
          await new Promise(resolve => setTimeout(resolve, 80));
          return { task: 'C', result: 'completed' };
        },
        timeout_ms: 30000
      }
    ],
    parallel: true, // Execute in parallel
    on_failure: 'continue' as const, // Continue on failure
    timeout_ms: 60000
  };

  const startTime = Date.now();

  const result = await workflowEngine.execute(
    parallelWorkflow,
    { batch_id: 'batch-1' }
  );

  const duration = Date.now() - startTime;

  logger.info('Parallel workflow completed', {
    duration_ms: duration,
    state: result.state,
    steps_completed: result.steps_completed
  });

  // With parallel execution, total time should be ~max(task times) not sum
  // Expected: ~150ms (slowest task) not 330ms (sum of all tasks)
}

// ============================================================================
// Run All Examples
// ============================================================================

async function runAllExamples() {
  logger.info('='.repeat(60));
  logger.info('Orchestration Layer Examples');
  logger.info('='.repeat(60));

  try {
    await example1_BasicEvents();
    await new Promise(resolve => setTimeout(resolve, 100));

    await example2_RAGPipeline();
    await new Promise(resolve => setTimeout(resolve, 100));

    await example3_WorkflowExecution();
    await new Promise(resolve => setTimeout(resolve, 100));

    await example4_CustomWorkflow();
    await new Promise(resolve => setTimeout(resolve, 100));

    await example5_ErrorHandling();
    await new Promise(resolve => setTimeout(resolve, 100));

    await example6_CorrelationTracking();
    await new Promise(resolve => setTimeout(resolve, 100));

    await example7_ParallelWorkflow();

    logger.info('='.repeat(60));
    logger.info('All examples completed successfully');
    logger.info('='.repeat(60));
  } catch (error) {
    logger.error('Example execution failed', error as Error);
  }
}

// Run examples if this file is executed directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}

export {
  example1_BasicEvents,
  example2_RAGPipeline,
  example3_WorkflowExecution,
  example4_CustomWorkflow,
  example5_ErrorHandling,
  example6_CorrelationTracking,
  example7_ParallelWorkflow
};
