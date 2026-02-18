/**
 * Basic Usage Examples for OpenEvolve Integration Library
 */

import {
  OpenEvolveClient,
  IntegrationName,
  DecompositionInputs,
  LeanAideInputs,
  EvolutionInputs,
  KnowledgeInputs,
  MakerInputs,
  CrewAIInputs
} from '@openevolve/integration-library';

// ============================================================================
// Initialization
// ============================================================================

import { createMockClient } from '@openevolve/integration-library/testing';

// Basic initialization
const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000'
});

// Mock client for testing
const mockClient = createMockClient({
  leanaide: { result: 'Theorem proved!', metadata: { executionTime: 100 } }
});

// With API key
const clientWithAuth = new OpenEvolveClient({
  baseUrl: 'https://api.openevolve.org',
  apiKey: process.env.OPENEVOLVE_API_KEY
});

// With custom configuration
const customClient = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000',
  timeout: 60000,
  retryAttempts: 5, // Changed from retries to retryAttempts
  debug: true,
  headers: {
    'X-Custom-Header': 'value'
  }
});

// ============================================================================
// Decomposition Examples
// ============================================================================

async function decomposeProblem() {
  const result = await client.integrations.decomposition.execute({
    operation: 'decompose',
    input: {
      problem: 'Build a scalable microservices architecture',
      strategy: 'hierarchical',
      options: { max_depth: 3 }
    }
  });

  // Result structure depends on backend response
  console.log('Decomposition result:', result);

  return result;
}

// ============================================================================
// LeanAide Examples
// ============================================================================

// Formal verification
async function verifyProof() {
  const result = await client.integrations.leanaide.execute({
    operation: 'prove',
    input: {
      theorem: 'forall n : Nat, n + 0 = n',
      strategy: 'auto',
      tactics: ['induction', 'rewrite', 'simp']
    }
  });

  console.log('Proof result:', result);

  return result;
}

// MCTS planning
async function mctsPlanning() {
  const result = await client.integrations.leanaide.execute({
    operation: 'mcts',
    input: {
      problem: 'Find optimal path in graph',
      config: { iterations: 5000 }
    }
  });

  console.log('MCTS result:', result);

  return result;
}

// ============================================================================
// Evolution Examples
// ============================================================================

async function evolutionaryAlgorithm() {
  const result = await client.integrations.evolution.execute({
    operation: 'evolution',
    config: {
      initial_population: [
        { params: [1, 2, 3] },
        { params: [4, 5, 6] }
      ],
      fitness_function: 'maximize_accuracy',
      generations: 100,
      mutation_rate: 0.1,
      crossover_rate: 0.8
    }
  });

  console.log('Evolution result:', result);

  return result;
}

async function adversarialEvolution() {
  const result = await client.integrations.evolution.execute({
    operation: 'adversarial',
    config: {
      initial_population: [{ model: 'baseline' }],
      fitness_function: 'adversarial_robustness',
      generations: 50
    }
  });

  console.log('Adversarial result:', result);
  return result;
}

// ============================================================================
// Knowledge Engine Examples
// ============================================================================

async function extractKnowledge() {
  const result = await client.integrations.knowledge.execute({
    operation: 'extract',
    input: {
      document: 'OpenAI is a research organization...',
      documentType: 'text'
    }
  });

  console.log('Extraction result:', result);

  return result;
}

async function queryKnowledge() {
  const result = await client.integrations.knowledge.execute({
    operation: 'query',
    input: {
      query: 'Find all organizations related to AI research',
      graph_id: 'graph-123'
    }
  });

  console.log('Query results:', result);
  return result;
}

// ============================================================================
// Maker Engine Examples
// ============================================================================

async function createTool() {
  const result = await client.integrations.maker.execute({
    operation: 'create',
    input: {
      name: 'DataProcessor',
      description: 'Processes and validates data',
      inputs: [
        { name: 'data', type: 'object', required: true }
      ],
      outputs: [
        { name: 'processed_data', type: 'object', required: true }
      ],
      logic: 'return data.filter(item => item.valid)'
    }
  });

  console.log('Created tool:', result);
  return result;
}

async function executeTool() {
  const result = await client.integrations.maker.execute({
    operation: 'execute',
    input: {
      toolId: 'tool-123',
      parameters: { data: [] }
    }
  });

  console.log('Execution result:', result);
  return result;
}

// ============================================================================
// CrewAI Examples
// ============================================================================

async function delegateTask() {
  const result = await client.integrations.crewai.execute({
    operation: 'delegate',
    input: {
      task: 'Analyze dataset and generate report',
      agent_type: 'specialist',
      constraints: {
        max_time: 3600,
        resources: ['cpu', 'memory']
      }
    }
  });

  console.log('Delegation result:', result);

  return result;
}

// ============================================================================
// Advanced Usage
// ============================================================================

// Batch execution
async function batchExecute() {
  const results = await client.executeBatch([
    {
      integration: IntegrationName.DECOMPOSITION,
      id: 'req1',
      inputs: {
        operation: 'decompose',
        input: { problem: 'Problem 1' }
      }
    },
    {
      integration: IntegrationName.DECOMPOSITION,
      id: 'req2',
      inputs: {
        operation: 'decompose',
        input: { problem: 'Problem 2' }
      }
    },
    {
      integration: IntegrationName.EVOLUTION,
      id: 'req3',
      inputs: {
        operation: 'evolution',
        config: {
          initial_population: [],
          fitness_function: 'accuracy',
          generations: 50
        }
      }
    }
  ]);

  console.log('Batch results:', results);
  return results;
}

// Health check
async function checkHealth() {
  const health = await client.healthCheck();
  console.log('Health status:', health);
  
  return health;
}

// Update configuration
async function updateConfig() {
  client.updateRetryConfig({
    maxAttempts: 5,
    maxDelay: 20000
  });
}

// ============================================================================
// Error Handling
// ============================================================================

async function withErrorHandling() {
  try {
    const result = await client.integrations.decomposition.execute({
      operation: 'decompose',
      input: { problem: 'Test problem' }
    });
    return result;
  } catch (error: any) {
    if (error.name === 'ValidationError') {
      console.error('Validation failed:', error.details);
    } else if (error.name === 'NetworkError') {
      console.error('Network error:', error.message);
    } else if (error.name === 'ExecutionError') {
      console.error('Execution failed:', error.details);
    }
    throw error;
  }
}

// ============================================================================
// Streaming
// ============================================================================

async function streamingExecution() {
  const result = await client.integrations.decomposition.executeStream(
    {
      operation: 'decompose',
      input: { problem: 'Complex problem' }
    },
    (update) => {
      console.log(`Progress: ${update.progress}%`);
      console.log(`Status: ${update.message}`);
      if (update.data) {
        console.log('Partial result:', update.data);
      }
    }
  );

  console.log('Final result:', result);
  return result;
}

// ============================================================================
// Export examples
// ============================================================================

export {
  decomposeProblem,
  verifyProof,
  mctsPlanning,
  evolutionaryAlgorithm,
  adversarialEvolution,
  extractKnowledge,
  queryKnowledge,
  createTool,
  executeTool,
  delegateTask,
  batchExecute,
  checkHealth,
  updateConfig,
  withErrorHandling,
  streamingExecution
};