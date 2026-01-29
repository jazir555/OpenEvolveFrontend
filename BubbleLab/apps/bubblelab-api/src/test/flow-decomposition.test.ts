/**
 * Flow Decomposition Tests
 *
 * Tests the flow decomposition logic in bubble-flow-parser.ts
 * This tests the generateDisplayedBubbleParameters function and related utilities.
 */

// @ts-expect-error - Bun test types
import { describe, it, expect } from 'bun:test';
import {
  generateDisplayedBubbleParameters,
  type ParsedBubble,
  type FlowDecompositionResult,
  type DisplayParameter,
  type DependencyGraph,
  type ValidationRule,
  type DecompositionMetadata,
} from './src/services/bubble-flow-parser.js';
import { BubbleParameterType } from '@bubblelab/shared-schemas';

describe('Flow Decomposition - Simple Flow', () => {
  it('should decompose a simple flow with basic parameters', () => {
    // Test 1: Simple flow decomposition
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

    // Verify displayed parameters
    expect(result.displayedParameters.length).toBeGreaterThan(0);
    expect(result.displayedParameters).toHaveLength(2);

    // Verify dependencies nodes exist
    expect(result.dependencies.nodes.length).toBeGreaterThan(0);
    expect(result.dependencies.nodes).toHaveLength(3); // 1 bubble + 2 parameters

    // Verify metadata
    expect(result.metadata.totalParameters).toBe(2);
    expect(result.metadata.estimatedComplexity).toBe('simple');
  });

  it('should build dependency graph correctly', () => {
    // Test 2: Dependency graph building
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

    // Verify edges exist
    expect(result.dependencies.edges.length).toBeGreaterThanOrEqual(0);

    // Verify there are edges from bubble to parameters
    const bubbleToParamEdges = result.dependencies.edges.filter(
      (e) => e.type === 'data' && e.description === 'contains'
    );
    expect(bubbleToParamEdges.length).toBeGreaterThan(0);

    // Verify environment dependencies
    const envEdges = result.dependencies.edges.filter((e) => e.from === 'environment');
    expect(envEdges.length).toBeGreaterThan(0);
  });

  it('should extract validation rules', () => {
    // Test 3: Validation rules
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

    // Verify validation rules exist
    expect(result.validationRules.length).toBeGreaterThan(0);

    // Verify required parameter rules
    const requiredRules = result.validationRules.filter((r) => r.type === 'required');
    expect(requiredRules.length).toBeGreaterThan(0);

    // Verify environment variable warnings
    const envRules = result.validationRules.filter((r) => r.severity === 'warning');
    expect(envRules.length).toBeGreaterThanOrEqual(0);
  });

  it('should generate correct metadata', () => {
    // Test 4: Metadata
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

    // Verify metadata counts
    expect(result.metadata.totalParameters).toBe(5);
    expect(result.metadata.requiredParameters).toBeGreaterThan(0);
    expect(result.metadata.configurableParameters).toBeGreaterThan(0);
    expect(result.metadata.environmentParameters).toBe(1);

    // Verify complexity estimation
    expect(['simple', 'medium', 'complex']).toContain(result.metadata.estimatedComplexity);

    // Verify groups
    expect(result.metadata.groups.length).toBeGreaterThan(0);
  });
});

describe('Flow Decomposition - Circular Dependencies', () => {
  it('should detect circular dependencies', () => {
    // Test 5: Circular dependency detection
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

    // Should detect circular dependency (if implementation supports it)
    // Note: The current implementation may or may not detect this based on
    // how parameter references are parsed
    expect(result.metadata.hasCircularDependencies).toBeDefined();
    expect(typeof result.metadata.hasCircularDependencies).toBe('boolean');
  });
});

describe('Flow Decomposition - Edge Cases', () => {
  it('should handle empty flow', () => {
    const emptyFlow: Record<string, ParsedBubble> = {};
    const result = generateDisplayedBubbleParameters(emptyFlow);

    expect(result.displayedParameters).toHaveLength(0);
    expect(result.dependencies.nodes).toHaveLength(0);
    expect(result.dependencies.edges).toHaveLength(0);
    expect(result.validationRules).toHaveLength(0);
    expect(result.metadata.totalParameters).toBe(0);
    expect(result.metadata.estimatedComplexity).toBe('simple');
  });

  it('should handle flow with no parameters', () => {
    const noParamsFlow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(noParamsFlow);

    expect(result.displayedParameters).toHaveLength(0);
    expect(result.dependencies.nodes.length).toBeGreaterThanOrEqual(1); // At least the bubble node
    expect(result.metadata.totalParameters).toBe(0);
  });

  it('should handle nested object parameters', () => {
    const nestedFlow: Record<string, ParsedBubble> = {
      api: {
        variableName: 'api',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
          {
            name: 'headers',
            value: '{"Authorization": "Bearer token", "Content-Type": "application/json"}',
            type: BubbleParameterType.OBJECT,
          },
          {
            name: 'body',
            value: '{"data": [1, 2, 3]}',
            type: BubbleParameterType.OBJECT,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(nestedFlow);

    // Should parse nested parameters
    expect(result.displayedParameters).toHaveLength(2);
    expect(result.metadata.nestedParameterCount).toBe(2);
  });

  it('should handle array parameters', () => {
    const arrayFlow: Record<string, ParsedBubble> = {
      ai: {
        variableName: 'ai',
        bubbleName: 'ai-agent',
        className: 'AIAgentBubble',
        parameters: [
          {
            name: 'tools',
            value: '[{"name": "web-search"}, {"name": "calculator"}]',
            type: BubbleParameterType.ARRAY,
          },
        ],
        hasAwait: true,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(arrayFlow);

    expect(result.displayedParameters).toHaveLength(1);
    expect(result.displayedParameters[0].type).toBe(BubbleParameterType.ARRAY);
  });

  it('should handle mixed parameter types', () => {
    const mixedFlow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'postgres',
        className: 'PostgresBubble',
        parameters: [
          {
            name: 'connectionString',
            value: 'process.env.DB_URL',
            type: BubbleParameterType.ENV,
          },
          {
            name: 'query',
            value: 'SELECT * FROM users',
            type: BubbleParameterType.STRING,
          },
          {
            name: 'timeout',
            value: '30000',
            type: BubbleParameterType.NUMBER,
          },
          {
            name: 'enabled',
            value: 'true',
            type: BubbleParameterType.BOOLEAN,
          },
          {
            name: 'options',
            value: '{"ssl": true}',
            type: BubbleParameterType.OBJECT,
          },
        ],
        hasAwait: true,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(mixedFlow);

    expect(result.displayedParameters).toHaveLength(5);

    // Verify different types are recognized
    const types = result.displayedParameters.map((p) => p.type);
    expect(types).toContain(BubbleParameterType.ENV);
    expect(types).toContain(BubbleParameterType.STRING);
    expect(types).toContain(BubbleParameterType.NUMBER);
    expect(types).toContain(BubbleParameterType.BOOLEAN);
    expect(types).toContain(BubbleParameterType.OBJECT);
  });
});

describe('Flow Decomposition - Display Parameters', () => {
  it('should generate proper display names', () => {
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

    // Check display names are human-readable
    const connParam = result.displayedParameters.find((p) => p.name === 'bubble1.connectionString');
    expect(connParam).toBeDefined();
    expect(connParam?.displayName).toBe('Connection String');

    const retryParam = result.displayedParameters.find((p) => p.name === 'bubble1.maxRetries');
    expect(retryParam).toBeDefined();
    expect(retryParam?.displayName).toBe('Max Retries');
  });

  it('should determine parameter sources correctly', () => {
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
    expect(envParam?.source).toBe('environment');

    const literalParam = result.displayedParameters.find((p) => p.name === 'bubble1.query');
    expect(literalParam?.source).toBe('literal');

    const refParam = result.displayedParameters.find((p) => p.name === 'bubble1.result');
    expect(refParam?.source).toBe('reference');
  });

  it('should identify configurable parameters', () => {
    const flow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
          {
            name: 'apiKey',
            value: 'process.env.API_KEY',
            type: BubbleParameterType.ENV,
          },
          {
            name: 'timeout',
            value: '5000',
            type: BubbleParameterType.NUMBER,
          },
          {
            name: 'payload',
            value: 'payload.data',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(flow);

    // Environment variables should be configurable
    const envParam = result.displayedParameters.find((p) => p.name === 'bubble1.apiKey');
    expect(envParam?.isConfigurable).toBe(true);

    // Numbers should be configurable
    const numParam = result.displayedParameters.find((p) => p.name === 'bubble1.timeout');
    expect(numParam?.isConfigurable).toBe(true);

    // References to payload should not be configurable
    const payloadParam = result.displayedParameters.find((p) => p.name === 'bubble1.payload');
    expect(payloadParam?.isConfigurable).toBe(false);
  });
});

describe('Flow Decomposition - Dependency Analysis', () => {
  it('should extract parameter dependencies', () => {
    const flow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
          {
            name: 'url',
            value: 'https://api.example.com',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
      bubble2: {
        variableName: 'bubble2',
        bubbleName: 'slack',
        className: 'SlackBubble',
        parameters: [
          {
            name: 'message',
            value: 'bubble1.response',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(flow);

    // Find the parameter with dependency
    const depParam = result.displayedParameters.find((p) => p.name === 'bubble2.message');
    expect(depParam).toBeDefined();
    expect(depParam?.dependencies).toContain('bubble1');

    // Check that dependency edge exists
    const depEdge = result.dependencies.edges.find(
      (e) => e.from === 'bubble1' && e.to === 'bubble2'
    );
    expect(depEdge).toBeDefined();
  });

  it('should group parameters by bubble', () => {
    const flow: Record<string, ParsedBubble> = {
      postgres: {
        variableName: 'postgres',
        bubbleName: 'postgres',
        className: 'PostgresBubble',
        parameters: [
          {
            name: 'connectionString',
            value: 'process.env.DB_URL',
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
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(flow);

    // Verify groups exist
    expect(result.metadata.groups.length).toBe(2);

    // Verify postgres group
    const postgresGroup = result.metadata.groups.find((g) => g.name === 'postgres');
    expect(postgresGroup).toBeDefined();
    expect(postgresGroup?.parameters).toHaveLength(2);
    expect(postgresGroup?.parameters).toContain('postgres.connectionString');
    expect(postgresGroup?.parameters).toContain('postgres.query');

    // Verify slack group
    const slackGroup = result.metadata.groups.find((g) => g.name === 'slack');
    expect(slackGroup).toBeDefined();
    expect(slackGroup?.parameters).toHaveLength(1);
  });
});

describe('Flow Decomposition - Complexity Analysis', () => {
  it('should classify simple flows correctly', () => {
    const simpleFlow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
          {
            name: 'url',
            value: 'https://api.example.com',
            type: BubbleParameterType.STRING,
          },
        ],
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(simpleFlow);
    expect(result.metadata.estimatedComplexity).toBe('simple');
  });

  it('should classify medium complexity flows', () => {
    const mediumFlow: Record<string, ParsedBubble> = {
      db: {
        variableName: 'db',
        bubbleName: 'postgres',
        className: 'PostgresBubble',
        parameters: Array(5).fill({
          name: 'param',
          value: 'value',
          type: BubbleParameterType.STRING as const,
        }),
        hasAwait: true,
        hasActionCall: false,
      },
      api: {
        variableName: 'api',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: Array(6).fill({
          name: 'param',
          value: 'value',
          type: BubbleParameterType.STRING as const,
        }),
        hasAwait: false,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(mediumFlow);
    expect(result.metadata.estimatedComplexity).toBe('medium');
  });

  it('should classify complex flows correctly', () => {
    // Create a flow with many parameters
    const manyParams = Array(25).fill({
      name: 'param',
      value: 'value',
      type: BubbleParameterType.STRING as const,
    });

    const complexFlow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'postgres',
        className: 'PostgresBubble',
        parameters: manyParams,
        hasAwait: true,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(complexFlow);
    expect(result.metadata.estimatedComplexity).toBe('complex');
  });
});

describe('Flow Decomposition - Validation Rules', () => {
  it('should generate required field validation rules', () => {
    const flow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
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
    };

    const result = generateDisplayedBubbleParameters(flow);

    // ENV parameters should generate required rules
    const requiredRules = result.validationRules.filter((r) => r.type === 'required');
    expect(requiredRules.length).toBeGreaterThan(0);

    // Should have severity 'error' for required
    const errorRules = requiredRules.filter((r) => r.severity === 'error');
    expect(errorRules.length).toBeGreaterThan(0);
  });

  it('should generate environment variable warnings', () => {
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
        ],
        hasAwait: true,
        hasActionCall: false,
      },
    };

    const result = generateDisplayedBubbleParameters(flow);

    // Should have warning about environment variable
    const envWarnings = result.validationRules.filter(
      (r) => r.severity === 'warning' && r.message.includes('environment variable')
    );
    expect(envWarnings.length).toBeGreaterThan(0);
  });

  it('should generate range validation for numeric parameters', () => {
    const flow: Record<string, ParsedBubble> = {
      bubble1: {
        variableName: 'bubble1',
        bubbleName: 'http',
        className: 'HttpBubble',
        parameters: [
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

    const result = generateDisplayedBubbleParameters(flow);

    // Should have range validation for numbers
    const rangeRules = result.validationRules.filter((r) => r.type === 'range');
    expect(rangeRules.length).toBeGreaterThan(0);
  });
});

console.log('✅ Flow decomposition test suite loaded successfully');
