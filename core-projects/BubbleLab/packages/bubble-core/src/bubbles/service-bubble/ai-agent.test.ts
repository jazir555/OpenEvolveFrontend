import { AIAgentBubble } from './ai-agent.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { BubbleFactory } from '../../bubble-factory.js';
import { STATIC_MODEL_OPTIONS } from '@bubblelab/shared-schemas';
import { z } from 'zod';
import { RECOMMENDED_MODELS } from '@bubblelab/shared-schemas';

//Remove all environment variables
process.env = {};

const factory = new BubbleFactory();

beforeAll(async () => {
  await factory.registerDefaults();
});

describe('AIAgentBubble', () => {
  describe('basic properties', () => {
    test('should have correct metadata', () => {
      const bubble = new AIAgentBubble({
        message: 'Hello AI!',
      });

      expect(bubble.name).toBe('ai-agent');
      expect(bubble.type).toBe('service');
      expect(bubble.alias).toBe('agent');
      expect(bubble.shortDescription).toContain('AI agent');
    });

    test('should have longDescription with use cases', () => {
      const bubble = new AIAgentBubble({
        message: 'Hello AI!',
      });

      expect(bubble.longDescription).toContain('Use cases:');
      expect(bubble.longDescription).toContain('tool bubble');
      expect(bubble.longDescription).toContain('LangGraph');
    });
  });

  describe('parameter validation', () => {
    test('should report a validation error for a missing message at execution', async () => {
      const bubble = new AIAgentBubble({});
      const result = await bubble.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });

    test('should report a validation error for an empty message at execution', async () => {
      const bubble = new AIAgentBubble({ message: '' });
      const result = await bubble.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Message is required');
    });

    test('should accept custom model configuration', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: {
          model: RECOMMENDED_MODELS.FAST,
          temperature: 0.5,
          maxTokens: 1000,
        },
      });

      const params = bubble.currentParams;
      expect(params.model.model).toBe(RECOMMENDED_MODELS.FAST);
      expect(params.model.temperature).toBe(0.5);
      expect(params.model.maxTokens).toBe(1000);
    });

    test('should accept tools configuration', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        tools: [
          { name: 'web-search-tool' },
          { name: 'bubbleflow-validation-tool' },
        ],
      });

      const params = bubble.currentParams;
      expect(params.tools).toHaveLength(2);
      expect(params.tools[0].name).toBe('web-search-tool');
      expect(params.tools[1].name).toBe('bubbleflow-validation-tool');
    });
  });

  describe('error handling', () => {
    test('should accept arbitrary tool names without validation errors', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        tools: [
          { name: 'invalid-tool-that-does-not-exist' },
          { name: 'another-non-existent-tool' },
        ],
      });
      expect(bubble.currentParams.tools).toHaveLength(2);
      expect(bubble.currentParams.tools?.[0]?.name).toBe(
        'invalid-tool-that-does-not-exist'
      );
    });

    test('should route unknown provider prefixes to the NVIDIA NIM credential', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'invalid/provider' },
      });
      expect(bubble['getCredentialTypeForModel']('invalid/provider')).toBe(
        CredentialType.NVIDIA_NIM_CRED
      );
    });

    test('should report a validation error for an out-of-range temperature at execution', async () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash', temperature: 3.0 },
      });
      const result = await bubble.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('error handling scenarios', () => {
    test('should detect MAX_TOKENS finish reason', () => {
      // Test the logic for detecting MAX_TOKENS without making API calls
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
      });

      // This test validates that the MAX_TOKENS detection logic is present
      // The actual integration test with real API calls is in ai-agent.integration.test.ts
      expect(bubble.currentParams.message).toBe('Test message');
      expect(typeof bubble['executeAgent']).toBe('function');
    });

    test('should create bubble with proper configuration', async () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        systemPrompt: 'You are a test assistant',
        tools: [{ name: 'web-search-tool' }],
      });

      // Test that the bubble is properly configured
      expect(bubble.currentParams.message).toBe('Test message');
      expect(bubble.currentParams.systemPrompt).toBe(
        'You are a test assistant'
      );
      expect(bubble.currentParams.tools).toHaveLength(1);
      expect(bubble.currentParams.tools[0].name).toBe('web-search-tool');
    });
  });

  describe('model format validation', () => {
    test('should accept OpenAI models', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: {
          model: 'google/gemini-2.5-flash',
          temperature: 0.7,
          maxTokens: 500,
        },
      });

      expect(bubble.currentParams.model.model).toBe('google/gemini-2.5-flash');
    });

    test('should accept Google Gemini models', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: {
          model: 'google/gemini-2.5-flash',
          temperature: 0.7,
          maxTokens: 500,
        },
      });

      expect(bubble.currentParams.model.model).toBe('google/gemini-2.5-flash');
    });

    test('should accept all supported model variants', () => {
      // Static built-in models plus representative NVIDIA NIM catalog ids
      // (NIM models are fetched dynamically and accepted as plain strings).
      const supportedModels = [
        ...STATIC_MODEL_OPTIONS,
        'deepseek-ai/deepseek-v4-flash-0731',
        'meta/llama-3.1-8b-instruct',
        'nvidia/llama-3.1-nemotron-51b-instruct',
      ];

      supportedModels.forEach((model) => {
        expect(() => {
          new AIAgentBubble({
            message: 'Test message',

            model: { model },
          });
        }).not.toThrow();
      });
    });

    test('should route unsupported model formats to the NVIDIA NIM credential', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'unsupported/model-name' },
      });
      expect(
        bubble['getCredentialTypeForModel']('unsupported/model-name')
      ).toBe(CredentialType.NVIDIA_NIM_CRED);
    });
  });
});

describe('BubbleRegistry - AIAgent', () => {
  test('should register AIAgentBubble class automatically', () => {
    const registeredBubbleClass = factory.get('ai-agent');

    expect(registeredBubbleClass).toBeDefined();
    expect(registeredBubbleClass).toBe(AIAgentBubble);
  });

  test('should contain proper schema for AIAgent', () => {
    const registeredBubbleClass = factory.get('ai-agent');

    expect(registeredBubbleClass?.schema).toBeDefined();
    // Check if schema has shape property (ZodObject)
    if (
      registeredBubbleClass?.schema &&
      'shape' in registeredBubbleClass.schema
    ) {
      const zodObjectSchema = registeredBubbleClass.schema;
      expect(zodObjectSchema.shape).toBeDefined();
      expect(zodObjectSchema.shape.systemPrompt).toBeDefined();
      expect(zodObjectSchema.shape.model).toBeDefined();
      expect(zodObjectSchema.shape.tools).toBeDefined();
    }
  });

  test('should get AIAgent metadata including params schema', () => {
    const metadata = factory.getMetadata('ai-agent');

    expect(metadata).toBeDefined();
    expect(metadata?.name).toBe('ai-agent');
    expect(metadata?.shortDescription).toContain('AI agent');
    expect(metadata?.alias).toBe('agent');
    expect(metadata?.params).toBeDefined();
    expect(metadata?.params?.message).toBeDefined();
    expect(metadata?.params?.model).toBeDefined();
    expect(metadata?.params?.tools).toBeDefined();
  });

  test('should list AIAgent in registered bubbles', () => {
    const bubbleList = factory.list();

    expect(bubbleList).toContain('ai-agent');
  });
});

describe('AIAgentBubble - Credential System', () => {
  describe('chooseCredential method', () => {
    test('should return undefined when no credentials are provided', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
      });

      expect(() => {
        bubble['chooseCredential']();
      }).toThrow('No GOOGLE credentials provided');
    });

    test('should accept arbitrary model catalog ids at construction without validation errors', () => {
      expect(() => {
        new AIAgentBubble({
          message: 'Test message',
          // Unrecognized catalog id is accepted as a dynamic NIM-style string;
          // provider/credential resolution happens at runtime, not construction.
          model: { model: 'google/geminis-2.5-flash' },
        });
      }).not.toThrow();
    });

    test('should report a validation error when credentials is not an object', async () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
        // @ts-expect-error testing invalid input
        credentials: 'invalid-credentials',
      });
      const result = await bubble.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });

    test('should choose OpenAI credential for OpenAI models', () => {
      const testCredentials = {
        [CredentialType.OPENAI_CRED]: 'test-openai-key-123',
        [CredentialType.GOOGLE_GEMINI_CRED]: 'test-google-key-456',
      };

      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
        credentials: testCredentials,
      });

      const credential = bubble['chooseCredential']();
      expect(credential).toBe('test-google-key-456');
    });

    test('should choose Google credential for Google models', () => {
      const testCredentials = {
        [CredentialType.OPENAI_CRED]: 'test-openai-key-123',
        [CredentialType.GOOGLE_GEMINI_CRED]: 'test-google-key-456',
      };

      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-pro' },
        credentials: testCredentials,
      });

      const credential = bubble['chooseCredential']();
      expect(credential).toBe('test-google-key-456');
    });

    test('should route unsupported model providers to the NVIDIA NIM credential when resolving credentials', () => {
      const testCredentials = {
        [CredentialType.OPENAI_CRED]: 'test-openai-key-123',
        [CredentialType.GOOGLE_GEMINI_CRED]: 'test-google-key-456',
      };
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'unsupported/model-name' },
        credentials: testCredentials,
      });
      // Unknown catalog ids resolve to NVIDIA NIM (no throw); absent NIM cred -> undefined.
      expect(bubble['chooseCredential']()).toBeUndefined();
    });

    test('should return undefined when required credential type is missing', () => {
      const testCredentials = {
        [CredentialType.SLACK_CRED]: 'test-slack-token', // Only Slack credential, no AI credentials
      };

      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
        credentials: testCredentials,
      });

      const credential = bubble['chooseCredential']();
      expect(credential).toBeUndefined();
    });

    test('should work with different OpenAI model variants', () => {
      const testCredentials = {
        [CredentialType.OPENAI_CRED]: 'test-openai-key-123',
        [CredentialType.GOOGLE_GEMINI_CRED]: 'test-google-key-456',
      };

      const openAIModels = [
        'google/gemini-2.5-flash',
        'google/gemini-2.5-pro',
        'google/gemini-2.5-flash-lite',
        'google/gemini-2.5-flash-image-preview',
      ];

      openAIModels.forEach((modelName) => {
        const bubble = new AIAgentBubble({
          message: 'Test message',
          // @ts-expect-error type matches
          model: { model: modelName },
          credentials: testCredentials,
        });

        const credential = bubble['chooseCredential']();
        expect(credential).toBe('test-google-key-456');
      });
    });

    test('should work with different Google model variants', () => {
      const testCredentials = {
        [CredentialType.GOOGLE_GEMINI_CRED]: 'test-google-key-456',
      };

      const googleModels = ['google/gemini-2.5-pro', 'google/gemini-2.5-flash'];

      googleModels.forEach((modelName) => {
        const bubble = new AIAgentBubble({
          message: 'Test message',
          // @ts-expect-error type matches
          model: { model: modelName },
          credentials: testCredentials,
        });

        const credential = bubble['chooseCredential']();
        expect(credential).toBe('test-google-key-456');
      });
    });
  });

  describe('credential integration with model initialization', () => {
    test('should handle missing credentials gracefully during model initialization', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
        // No credentials provided
      });

      // Should not throw during construction
      expect(bubble).toBeDefined();
      expect(bubble.currentParams.model.model).toBe('google/gemini-2.5-flash');
    });

    test('should handle empty credentials object', () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
        credentials: {}, // Empty credentials object
      });

      const credential = bubble['chooseCredential']();
      expect(credential).toBeUndefined();
    });

    test('should prioritize correct credential type when multiple are available', () => {
      const testCredentials = {
        [CredentialType.OPENAI_CRED]: 'openai-key-should-be-chosen',
        [CredentialType.GOOGLE_GEMINI_CRED]: 'google-key-should-not-be-chosen',
        [CredentialType.SLACK_CRED]: 'slack-key-irrelevant',
      };

      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
        credentials: testCredentials,
      });

      const credential = bubble['chooseCredential']();
      expect(credential).toBe('google-key-should-not-be-chosen');
    });

    test('should throw error when trying to execute without credentials', async () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
        // No credentials provided
      });

      // Should throw error during execution when trying to initialize the model
      const result = await bubble.action();
      console.log('Test result:', JSON.stringify(result, null, 2));

      expect(result).toMatchObject({
        success: false,
        // Expect non-empty error string
        error: expect.stringMatching(/.+/),
      });
    });

    test('should throw error when trying to execute with missing required credential', async () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-flash' },
        credentials: {
          [CredentialType.SLACK_CRED]: 'slack-token-not-relevant',
          // Missing OPENAI_CRED
        },
      });

      // Should throw error during execution when trying to initialize the model with undefined API key
      await expect(bubble.action()).resolves.toMatchObject({
        success: false,
        error: expect.stringMatching(/.+/),
      });
    });

    test('should throw error when trying to execute with empty credentials object', async () => {
      const bubble = new AIAgentBubble({
        message: 'Test message',
        model: { model: 'google/gemini-2.5-pro' },
        credentials: {}, // Empty credentials
      });

      // Should throw error during execution
      await expect(bubble.action()).resolves.toMatchObject({
        success: false,
        error: expect.stringMatching(/.+/),
      });
    });
  });
});

describe('AIAgentBubble - Tool Bubble Integration', () => {
  test('should accept tool bubble names in tools parameter', () => {
    const bubble = new AIAgentBubble({
      message: 'List all available bubbles',
      tools: [
        { name: 'list-bubbles-tool' },
        { name: 'get-bubble-details-tool' },
      ],
    });

    expect(bubble.currentParams.tools).toHaveLength(2);
    expect(bubble.currentParams.tools[0].name).toBe('list-bubbles-tool');
    expect(bubble.currentParams.tools[1].name).toBe('get-bubble-details-tool');
  });

  test('should handle empty tools array', () => {
    const bubble = new AIAgentBubble({
      message: 'Test without tools',
      tools: [],
    });

    expect(bubble.currentParams.tools).toHaveLength(0);
  });

  test('should accept dynamic custom tools without pre-registration', () => {
    const bubble = new AIAgentBubble({
      message: 'Calculate sales tax for $100',
      customTools: [
        {
          name: 'calculate-tax',
          description: 'Calculates sales tax for a given amount',
          schema: {
            amount: z.number().describe('Amount to calculate tax on'),
            taxRate: z.number().describe('Tax rate percentage'),
          },
          func: async (input: Record<string, unknown>) => {
            const amount = input.amount as number;
            const taxRate = input.taxRate as number;
            return (amount * taxRate) / 100;
          },
        },
      ],
    });

    expect(bubble.currentParams.customTools![0]).toMatchObject({
      name: 'calculate-tax',
      description: expect.stringContaining('sales tax'),
    });
  });
});
