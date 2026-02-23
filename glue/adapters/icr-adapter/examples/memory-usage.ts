/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Memory Integration Example
 *
 * Demonstrates how to use ICR's Contextual Mode with Graphiti memory integration.
 * This example shows:
 * 1. Setting up the memory agent
 * 2. Making memory-enhanced contextual requests
 * 3. Retrieving and storing memories
 * 4. Learning from session outcomes
 */

import { GraphitiAdapter } from '@openevolve/graphiti-adapter';
import { ICRAdapter } from '../src/adapter';
import {
  EnhancedICRMemoryAgent,
  MemoryAgentConfig
} from '../src/memory/memory-agent';
import {
  ContextualSession,
  SessionOutcome,
  RefinementInsights
} from '../src/memory/canonical';

// ============================================================================
// CONFIGURATION
// ============================================================================

// Required: Set these environment variables before running
const config = {
  // ICR API
  ICR_API_URL: process.env.OPENEVOLVE_ICR_API_URL || 'http://localhost:8000',
  TIMEOUT_MS: parseInt(process.env.TIMEOUT_MS || '30000', 10),

  // Graphiti
  GRAPHITI_API_URL: process.env.GRAPHITI_API_URL || 'http://localhost:8080',
  NEO4J_URI: process.env.NEO4J_URI || 'bolt://localhost:7687',
  NEO4J_USER: process.env.NEO4J_USER || 'neo4j',
  NEO4J_PASSWORD: process.env.NEO4J_PASSWORD || 'password'
};

// ============================================================================
// SETUP
// ============================================================================

async function setupMemoryAgent(): Promise<ICRAdapter> {
  console.log('Setting up ICR adapter with Graphiti memory integration...');

  // Create Graphiti adapter
  const graphitiAdapter = new GraphitiAdapter({
    graphiti_api_url: config.GRAPHITI_API_URL,
    neo4j_uri: config.NEO4J_URI,
    neo4j_user: config.NEO4J_USER,
    neo4j_password: config.NEO4J_PASSWORD
  });

  // Wait for Graphiti to initialize
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Create memory agent configuration
  const memoryAgentConfig: MemoryAgentConfig = {
    graphiti: {
      graphitiAdapter,
      default_context_window: 5,
      max_historical_results: 20,
      enable_pattern_learning: true,
      enable_cross_session_learning: true
    },
    enable_historical_retrieval: true,
    enable_pattern_learning: true,
    enable_cross_session_learning: true,
    default_context_window: 5,
    min_relevance_score: 0.3,
    max_historical_results: 20,
    learning_threshold: 0.7,
    pattern_extraction_min_frequency: 2
  };

  // Create ICR adapter with memory
  const icrAdapter = new ICRAdapter({
    memoryAgentConfig
  });

  console.log('✓ ICR adapter with memory initialized');
  return icrAdapter;
}

// ============================================================================
// EXAMPLE 1: Memory-Enhanced Contextual Request
// ============================================================================

async function example1_MemoryEnhancedRequest(icrAdapter: ICRAdapter) {
  console.log('\n=== Example 1: Memory-Enhanced Contextual Request ===\n');

  const prompt = 'Refine this React component to improve performance and add memoization where appropriate';

  console.log('Prompt:', prompt);
  console.log('\nExecuting with memory enhancement...\n');

  try {
    const response = await icrAdapter.createContextualRequestWithMemory(
      prompt,
      {
        context_window: 5,
        enable_learning: true,
        temperature: 0.7
      }
    );

    console.log('✓ Request completed');
    console.log('\n--- Result ---');
    console.log('Success:', response.result.success);
    console.log('Iterations:', response.result.iteration_count);
    console.log('Execution Time:', response.result.execution_time_ms, 'ms');

    if (response.enriched_context) {
      console.log('\n--- Enriched Context ---');
      console.log('Historical Knowledge Items:', response.enriched_context.historical_knowledge.length);
      console.log('Related Patterns:', response.enriched_context.related_patterns.length);
      console.log('Suggested Approaches:', response.enriched_context.suggested_approaches.length);
      console.log('Common Pitfalls:', response.enriched_context.common_pitfalls.length);
      console.log('Confidence Score:', response.enriched_context.confidence_score);

      // Display top suggested approaches
      if (response.enriched_context.suggested_approaches.length > 0) {
        console.log('\nTop Suggested Approaches:');
        response.enriched_context.suggested_approaches.slice(0, 3).forEach((approach, i) => {
          console.log(`  ${i + 1}. ${approach}`);
        });
      }
    }

    if (response.learning_result) {
      console.log('\n--- Learning Result ---');
      console.log('Patterns Learned:', response.learning_result.patterns_learned);
      console.log('Patterns Updated:', response.learning_result.patterns_updated);
      console.log('New Relationships:', response.learning_result.new_relationships);
      console.log('Insights Extracted:', response.learning_result.insights_extracted);
    }
  } catch (error) {
    console.error('✗ Error:', error);
  }
}

// ============================================================================
// EXAMPLE 2: Direct Memory Agent Usage
// ============================================================================

async function example2_DirectMemoryUsage(memoryAgent: EnhancedICRMemoryAgent) {
  console.log('\n=== Example 2: Direct Memory Agent Usage ===\n');

  // Retrieve historical knowledge
  const query = 'React performance optimization with useMemo and useCallback';
  console.log('Query:', query);

  const enrichedContext = await memoryAgent.retrieveHistoricalKnowledge(
    query,
    10, // context window
    undefined // correlation ID (auto-generated)
  );

  console.log('\n✓ Historical knowledge retrieved');
  console.log('Knowledge Items:', enrichedContext.historical_knowledge.length);
  console.log('Related Patterns:', enrichedContext.related_patterns.length);
  console.log('Confidence Score:', enrichedContext.confidence_score);
  console.log('Processing Time:', enrichedContext.processing_time_ms, 'ms');

  // Display historical knowledge
  if (enrichedContext.historical_knowledge.length > 0) {
    console.log('\n--- Top Historical Knowledge ---');
    enrichedContext.historical_knowledge.slice(0, 3).forEach((knowledge, i) => {
      console.log(`\n${i + 1}. Session: ${knowledge.session_id.substring(0, 8)}...`);
      console.log(`   Pattern: ${knowledge.pattern_type}`);
      console.log(`   Outcome: ${knowledge.outcome}`);
      console.log(`   Insights: ${knowledge.insights.slice(0, 2).join('; ')}`);
    });
  }

  // Display related patterns
  if (enrichedContext.related_patterns.length > 0) {
    console.log('\n--- Related Patterns ---');
    enrichedContext.related_patterns.slice(0, 5).forEach((pattern, i) => {
      console.log(`\n${i + 1}. ${pattern.pattern_name}`);
      console.log(`   Type: ${pattern.pattern_type}`);
      console.log(`   Success Rate: ${(pattern.success_rate * 100).toFixed(1)}%`);
      console.log(`   Frequency: ${pattern.frequency}`);
    });
  }
}

// ============================================================================
// EXAMPLE 3: Manual Memory Storage
// ============================================================================

async function example3_ManualMemoryStorage(
  memoryAgent: EnhancedICRMemoryAgent,
  sessionId: string
) {
  console.log('\n=== Example 3: Manual Memory Storage ===\n');

  // Create refinement insights
  const refinementInsights: RefinementInsights = {
    session_id: sessionId,
    mode: 'contextual',
    iterations: [
      {
        session_id: sessionId,
        iteration_number: 1,
        refinement_type: 'quality_improvement',
        prompt: 'Optimize React component performance',
        content: 'Added useMemo and useCallback hooks',
        outcome: 'success',
        insights: [
          'Memoization reduced re-renders by 40%',
          'useCallback is effective for event handlers',
          'useMemo helps with expensive calculations'
        ],
        quality_metrics: {
          novelty_score: 0.6,
          quality_score: 0.9,
          improvement_percentage: 40
        },
        execution_time_ms: 1250,
        timestamp_utc: new Date().toISOString()
      }
    ],
    total_iterations: 1,
    successful_iterations: 1,
    failed_iterations: 0,
    total_execution_time_ms: 1250,
    average_quality_score: 0.9,
    overall_outcome: 'success',
    key_patterns_discovered: [
      'memoization_with_usememo',
      'callback_memoization_with_usecallback',
      'performance_monitoring'
    ],
    lessons_learned: [
      'Memoization is most effective for expensive computations',
      'Always profile before optimizing',
      'Over-memoization can hurt performance'
    ],
    session_start_utc: new Date(Date.now() - 2000).toISOString(),
    session_end_utc: new Date().toISOString()
  };

  console.log('Storing refinement insights...');

  await memoryAgent.storeRefinementInsights(
    refinementInsights,
    sessionId
  );

  console.log('✓ Refinement insights stored');
  console.log('Session ID:', sessionId);
  console.log('Iterations:', refinementInsights.total_iterations);
  console.log('Outcome:', refinementInsights.overall_outcome);
}

// ============================================================================
// EXAMPLE 4: Learning from Session Outcomes
// ============================================================================

async function example4_SessionLearning(
  memoryAgent: EnhancedICRMemoryAgent,
  sessionId: string
) {
  console.log('\n=== Example 4: Learning from Session Outcomes ===\n');

  // Create a contextual session
  const contextualSession: ContextualSession = {
    session_id: sessionId,
    mode: 'contextual',
    prompt: 'Build a React form with validation',
    agents_involved: ['main_generator', 'iterative_agent', 'memory_agent'],
    interactions: [
      {
        agent_type: 'main_generator',
        content: 'Generated initial form structure',
        timestamp_utc: new Date(Date.now() - 3000).toISOString(),
        execution_time_ms: 800
      },
      {
        agent_type: 'iterative_agent',
        content: 'Added validation logic to form fields',
        timestamp_utc: new Date(Date.now() - 2000).toISOString(),
        execution_time_ms: 650
      },
      {
        agent_type: 'memory_agent',
        content: 'Retrieved similar form patterns from memory',
        timestamp_utc: new Date(Date.now() - 1000).toISOString(),
        execution_time_ms: 200
      }
    ],
    context_window: 5,
    successes: 3,
    failures: 0,
    duration_ms: 1650,
    start_time_utc: new Date(Date.now() - 3000).toISOString(),
    end_time_utc: new Date().toISOString(),
    final_output: 'React form with comprehensive validation',
    quality_score: 0.85
  };

  // Create session outcomes
  const sessionOutcomes: SessionOutcome[] = [
    {
      session_id: sessionId,
      outcome: 'success',
      quality_score: 0.85,
      user_satisfaction: 0.9,
      iteration_count: 3,
      success_metrics: {
        execution_time_ms: 1650,
        agent_count: 3,
        interaction_count: 3
      },
      failure_reasons: [],
      successful_patterns: [
        'form_validation_pattern',
        'memory_retrieval_pattern',
        'iterative_refinement_pattern'
      ],
      problematic_patterns: [],
      lessons_learned: [
        'Memory retrieval accelerates development',
        'Iterative refinement improves quality',
        'Multi-agent collaboration is effective'
      ],
      timestamp_utc: new Date().toISOString()
    }
  ];

  console.log('Learning from session outcomes...');
  console.log('Session ID:', sessionId);
  console.log('Outcomes:', sessionOutcomes.length);

  const learningResult = await memoryAgent.learnFromSession(
    contextualSession,
    sessionOutcomes
  );

  console.log('\n✓ Learning completed');
  console.log('Patterns Learned:', learningResult.patterns_learned);
  console.log('Patterns Updated:', learningResult.patterns_updated);
  console.log('New Relationships:', learningResult.new_relationships);
  console.log('Insights Extracted:', learningResult.insights_extracted);
  console.log('Confidence Score:', learningResult.confidence_score?.toFixed(2));
  console.log('Processing Time:', learningResult.processing_time_ms, 'ms');
}

// ============================================================================
// EXAMPLE 5: Pattern Analysis
// ============================================================================

async function example5_PatternAnalysis(memoryAgent: EnhancedICRMemoryAgent) {
  console.log('\n=== Example 5: Pattern Analysis ===\n');

  // Create sample sessions for analysis
  const sessions: ContextualSession[] = [
    {
      session_id: 'session-1',
      mode: 'contextual',
      prompt: 'Build React component',
      agents_involved: ['main_generator', 'iterative_agent'],
      interactions: [],
      successes: 5,
      failures: 1,
      duration_ms: 2000,
      start_time_utc: new Date(Date.now() - 10000).toISOString(),
      end_time_utc: new Date(Date.now() - 8000).toISOString()
    },
    {
      session_id: 'session-2',
      mode: 'contextual',
      prompt: 'Optimize performance',
      agents_involved: ['iterative_agent', 'memory_agent'],
      interactions: [],
      successes: 3,
      failures: 0,
      duration_ms: 1500,
      start_time_utc: new Date(Date.now() - 7000).toISOString(),
      end_time_utc: new Date(Date.now() - 5500).toISOString()
    }
  ];

  console.log('Analyzing patterns across', sessions.length, 'sessions...');

  const patterns = await memoryAgent.analyzePatterns(sessions);

  console.log('\n✓ Pattern analysis completed');
  console.log('Patterns Discovered:', patterns.length);

  patterns.forEach((pattern, i) => {
    console.log(`\n${i + 1}. ${pattern.pattern_name}`);
    console.log(`   Type: ${pattern.pattern_type}`);
    console.log(`   Success Rate: ${(pattern.success_rate * 100).toFixed(1)}%`);
    console.log(`   Frequency: ${pattern.frequency}`);
    console.log(`   Related Sessions: ${pattern.related_sessions.length}`);
  });
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

async function main() {
  console.log('========================================');
  console.log('ICR Memory Integration Examples');
  console.log('========================================');

  try {
    // Setup
    const icrAdapter = await setupMemoryAgent();

    // Check if memory agent is available
    if (!icrAdapter.hasMemoryAgent()) {
      console.error('\n✗ Memory agent is not configured. Please check your configuration.');
      return;
    }

    // Get memory agent instance (requires private access or expose getter)
    // For this example, we'll use the adapter directly

    // Example 1: Memory-enhanced request
    await example1_MemoryEnhancedRequest(icrAdapter);

    // Note: Examples 2-5 require direct memory agent access
    // In production, expose the memory agent via a getter method

    console.log('\n========================================');
    console.log('Examples completed successfully!');
    console.log('========================================\n');
  } catch (error) {
    console.error('\n✗ Error:', error);
    process.exit(1);
  }
}

// Run examples if executed directly
if (require.main === module) {
  main().catch(console.error);
}

// Export for use as a module
export {
  setupMemoryAgent,
  example1_MemoryEnhancedRequest,
  example2_DirectMemoryUsage,
  example3_ManualMemoryStorage,
  example4_SessionLearning,
  example5_PatternAnalysis
};
