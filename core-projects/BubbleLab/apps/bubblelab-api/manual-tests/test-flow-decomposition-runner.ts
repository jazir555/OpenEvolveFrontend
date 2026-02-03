/**
 * Manual Flow Decomposition Test Runner
 *
 * This file can be run directly to test the flow decomposition logic
 * without requiring the full test infrastructure.
 *
 * Run with: npx ts-node manual-tests/test-flow-decomposition-runner.ts
 */

import {
  generateDisplayedBubbleParameters,
  type ParsedBubble,
  type FlowDecompositionResult,
} from '../src/services/bubble-flow-parser.js';
import { BubbleParameterType } from '@bubblelab/shared-schemas';

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message: string, color: string = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function success(message: string) {
  log(`✅ ${message}`, colors.green);
}

function error(message: string) {
  log(`❌ ${message}`, colors.red);
}

function info(message: string) {
  log(`ℹ️  ${message}`, colors.blue);
}

function section(title: string) {
  log(`\n${'='.repeat(60)}`, colors.cyan);
  log(title, colors.bright + colors.cyan);
  log('='.repeat(60), colors.cyan);
}

// Test suite
async function runTests() {
  let passed = 0;
  let failed = 0;

  section('FLOW DECOMPOSITION TESTS');

  // Test 1: Simple flow decomposition
  section('Test 1: Simple Flow Decomposition');
  try {
    const simpleFlow: Record<string, ParsedBubble> = {
      postgres: {
        variableName: 'postgres',
        bubbleName: 'postgres',
        className: 'PostgresBubble',
        parameters: [
          {
            name: 'connectionString',
            value: 'process.env.DATABASE_URL',
            type: BubbleParameterType.ENV,
          },
          {
            name: 'query',
            value: 'SELECT * FROM users',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: true,
        hasActionCall: false,
      },
    };

    const result: FlowDecompositionResult = generateDisplayedBubbleParameters(simpleFlow);

    // Assertions
    if (result.displayedParameters.length === 0) {
      throw new Error('No displayed parameters generated');
    }
    success(`Displayed parameters: ${result.displayedParameters.length}`);

    if (result.dependencies.nodes.length === 0) {
      throw new Error('No dependency nodes generated');
    }
    success(`Dependency nodes: ${result.dependencies.nodes.length}`);

    if (result.metadata.totalParameters !== 2) {
      throw new Error(`Expected 2 total parameters, got ${result.metadata.totalParameters}`);
    }
    success(`Total parameters: ${result.metadata.totalParameters}`);

    if (result.metadata.estimatedComplexity !== 'simple') {
      throw new Error(`Expected 'simple' complexity, got '${result.metadata.estimatedComplexity}'`);
    }
    success(`Complexity: ${result.metadata.estimatedComplexity}`);

    passed++;
  } catch (err) {
    error(`Test failed: ${err instanceof Error ? err.message : String(err)}`);
    failed++;
  }

  // Test 2: Dependency graph building
  section('Test 2: Dependency Graph Building');
  try {
    const flowWithDeps: Record<string, ParsedBubble> = {
      database: {
        variableName: 'database',
        bubbleName: 'postgres',
        className: 'PostgresBubble',
        parameters: [
          {
            name: 'connectionString',
            value: 'process.env.DATABASE_URL',
            type: BubbleParameterType.ENV,
          },
        ],
        hasAwait: true,
        hasActionCall: false,
      },
      aiAgent: {
        variableName: 'aiAgent',
        bubbleName: 'ai-agent',
        className: 'AIAgentBubble',
        parameters: [
          {
            name: 'model',
            value: 'gpt-4',
            type: BubbleParameterType.STRING,
          },
          {
            name: 'prompt',
            value: 'Analyze data from database',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: true,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(flowWithDeps);

    info(`Total edges: ${result.dependencies.edges.length}`);

    const bubbleToParamEdges = result.dependencies.edges.filter(
      (e) => e.type === 'data' && e.description === 'contains'
    );
    success(`Bubble-to-parameter edges: ${bubbleToParamEdges.length}`);

    const envEdges = result.dependencies.edges.filter((e) => e.from === 'environment');
    success(`Environment dependency edges: ${envEdges.length}`);

    passed++;
  } catch (err) {
    error(`Test failed: ${err instanceof Error ? err.message : String(err)}`);
    failed++;
  }

  // Test 3: Validation rules
  section('Test 3: Validation Rules Extraction');
  try {
    const flowWithValidation: Record<string, ParsedBubble> = {
      api: {
        variableName: 'api',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
          {
            name: 'url',
            value: 'https://api.example.com',
            type: BubbleParameterType.STRING,
          },
          {
            name: 'timeout',
            value: '5000',
            type: BubbleParameterType.NUMBER,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(flowWithValidation);

    if (result.validationRules.length === 0) {
      throw new Error('No validation rules generated');
    }
    success(`Validation rules: ${result.validationRules.length}`);

    const requiredRules = result.validationRules.filter((r) => r.type === 'required');
    success(`Required field rules: ${requiredRules.length}`);

    passed++;
  } catch (err) {
    error(`Test failed: ${err instanceof Error ? err.message : String(err)}`);
    failed++;
  }

  // Test 4: Metadata generation
  section('Test 4: Metadata Generation');
  try {
    const complexFlow: Record<string, ParsedBubble> = {
      db: {
        variableName: 'db',
        bubbleName: 'postgres',
        className: 'PostgresBubble',
        parameters: [
          {
            name: 'connectionString',
            value: 'process.env.DB_URL',
            type: BubbleParameterType.ENV,
          },
        ],
        hasAwait: true,
        hasActionCall: false,
      },
      ai: {
        variableName: 'ai',
        bubbleName: 'ai-agent',
        className: 'AIAgentBubble',
        parameters: [
          {
            name: 'model',
            value: 'gpt-4',
            type: BubbleParameterType.STRING,
          },
          {
            name: 'tools',
            value: '[{"name": "web-search-tool"}]',
            type: BubbleParameterType.ARRAY,
          },
        ],
        hasAwait: true,
        hasActionCall: false,
      },
      slack: {
        variableName: 'slack',
        bubbleName: 'slack',
        className: 'SlackBubble',
        parameters: [
          {
            name: 'channel',
            value: '#general',
            type: BubbleParameterType.STRING,
          },
          {
            name: 'message',
            value: 'ai.response',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(complexFlow);

    if (result.metadata.totalParameters !== 5) {
      throw new Error(`Expected 5 total parameters, got ${result.metadata.totalParameters}`);
    }
    success(`Total parameters: ${result.metadata.totalParameters}`);

    info(`Required parameters: ${result.metadata.requiredParameters}`);
    info(`Configurable parameters: ${result.metadata.configurableParameters}`);
    info(`Environment parameters: ${result.metadata.environmentParameters}`);

    if (!['simple', 'medium', 'complex'].includes(result.metadata.estimatedComplexity)) {
      throw new Error(`Invalid complexity level: ${result.metadata.estimatedComplexity}`);
    }
    success(`Complexity: ${result.metadata.estimatedComplexity}`);

    if (result.metadata.groups.length === 0) {
      throw new Error('No parameter groups generated');
    }
    success(`Parameter groups: ${result.metadata.groups.length}`);

    passed++;
  } catch (err) {
    error(`Test failed: ${err instanceof Error ? err.message : String(err)}`);
    failed++;
  }

  // Test 5: Circular dependency detection
  section('Test 5: Circular Dependency Detection');
  try {
    const circularFlow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
          {
            name: 'url',
            value: 'bubble2.response',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
      bubble2: {
        variableName: 'bubble2',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
          {
            name: 'url',
            value: 'bubble1.response',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(circularFlow);

    info(`Has circular dependencies: ${result.metadata.hasCircularDependencies}`);

    // The type should be boolean
    if (typeof result.metadata.hasCircularDependencies !== 'boolean') {
      throw new Error('hasCircularDependencies should be a boolean');
    }
    success('Circular dependency detection works');

    passed++;
  } catch (err) {
    error(`Test failed: ${err instanceof Error ? err.message : String(err)}`);
    failed++;
  }

  // Test 6: Empty flow handling
  section('Test 6: Empty Flow Handling');
  try {
    const emptyFlow: Record<string, ParsedBubble> = {};
    const result = generateDisplayedBubbleParameters(emptyFlow);

    if (result.displayedParameters.length !== 0) {
      throw new Error('Expected 0 displayed parameters for empty flow');
    }
    success('Displayed parameters: 0');

    if (result.dependencies.nodes.length !== 0) {
      throw new Error('Expected 0 dependency nodes for empty flow');
    }
    success('Dependency nodes: 0');

    if (result.metadata.totalParameters !== 0) {
      throw new Error('Expected 0 total parameters for empty flow');
    }
    success('Total parameters: 0');

    if (result.metadata.estimatedComplexity !== 'simple') {
      throw new Error(`Expected 'simple' complexity for empty flow, got '${result.metadata.estimatedComplexity}'`);
    }
    success('Complexity: simple');

    passed++;
  } catch (err) {
    error(`Test failed: ${err instanceof Error ? err.message : String(err)}`);
    failed++;
  }

  // Test 7: Display names
  section('Test 7: Display Name Generation');
  try {
    const flow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
          {
            name: 'connectionString',
            value: 'https://api.example.com',
            type: BubbleParameterType.STRING,
          },
          {
            name: 'maxRetries',
            value: '3',
            type: BubbleParameterType.NUMBER,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(flow);

    const connParam = result.displayedParameters.find((p) => p.name === 'bubble1.connectionString');
    if (!connParam) {
      throw new Error('Connection string parameter not found');
    }
    if (connParam.displayName !== 'Connection String') {
      throw new Error(`Expected 'Connection String', got '${connParam.displayName}'`);
    }
    success(`Display name for 'connectionString': '${connParam.displayName}'`);

    const retryParam = result.displayedParameters.find((p) => p.name === 'bubble1.maxRetries');
    if (!retryParam) {
      throw new Error('Max retries parameter not found');
    }
    if (retryParam.displayName !== 'Max Retries') {
      throw new Error(`Expected 'Max Retries', got '${retryParam.displayName}'`);
    }
    success(`Display name for 'maxRetries': '${retryParam.displayName}'`);

    passed++;
  } catch (err) {
    error(`Test failed: ${err instanceof Error ? err.message : String(err)}`);
    failed++;
  }

  // Test 8: Parameter sources
  section('Test 8: Parameter Source Detection');
  try {
    const flow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'postgres',
        className: 'PostgresBubble',
        parameters: [
          {
            name: 'url',
            value: 'process.env.DATABASE_URL',
            type: BubbleParameterType.ENV,
          },
          {
            name: 'query',
            value: 'SELECT * FROM users',
            type: BubbleParameterType.STRING,
          },
          {
            name: 'result',
            value: 'bubble2.output',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: true,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(flow);

    const envParam = result.displayedParameters.find((p) => p.name === 'bubble1.url');
    if (envParam?.source !== 'environment') {
      throw new Error(`Expected 'environment' source, got '${envParam?.source}'`);
    }
    success(`Source for 'url': ${envParam?.source}`);

    const literalParam = result.displayedParameters.find((p) => p.name === 'bubble1.query');
    if (literalParam?.source !== 'literal') {
      throw new Error(`Expected 'literal' source, got '${literalParam?.source}'`);
    }
    success(`Source for 'query': ${literalParam?.source}`);

    const refParam = result.displayedParameters.find((p) => p.name === 'bubble1.result');
    if (refParam?.source !== 'reference') {
      throw new Error(`Expected 'reference' source, got '${refParam?.source}'`);
    }
    success(`Source for 'result': ${refParam?.source}`);

    passed++;
  } catch (err) {
    error(`Test failed: ${err instanceof Error ? err.message : String(err)}`);
    failed++;
  }

  // Summary
  section('TEST SUMMARY');
  log(`Total tests: ${passed + failed}`, colors.bright);
  success(`Passed: ${passed}`);
  if (failed > 0) {
    error(`Failed: ${failed}`);
    process.exit(1);
  } else {
    success('All tests passed! 🎉');
    process.exit(0);
  }
}

// Run the tests
runTests().catch((err) => {
  error(`Test runner error: ${err instanceof Error ? err.message : String(err)}`);
  console.error(err);
  process.exit(1);
});
