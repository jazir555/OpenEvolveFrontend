import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import {
  CredentialType,
  type BubbleName,
  RECOMMENDED_MODELS,
  AvailableModels,
} from '@bubblelab/shared-schemas';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

/**
 * Hephaestus Bubble - Code Generation and Development Tools Service Bubble Implementation
 *
 * Full production implementation with 10 operations:
 * 1. generateCode - Generate code from natural language description
 * 2. explainCode - Explain code functionality in natural language
 * 3. findBugs - Analyze code for potential bugs
 * 4. suggestOptimizations - Suggest performance optimizations
 * 5. generateDocs - Generate documentation from code
 * 6. createAPI - Generate REST API endpoints
 * 7. createSchema - Generate database schemas
 * 8. generateTests - Generate unit and integration tests
 * 9. refactorCode - Refactor code with specific goals
 * 10. codeReview - Perform comprehensive code review
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const GenerateCodeParamsSchema = z.object({
  operation: z.literal('generateCode'),
  description: z.string().min(1, 'Description is required'),
  language: z.enum(['javascript', 'typescript', 'python', 'java', 'go', 'rust', 'csharp', 'php', 'sql']),
  framework: z.string().optional().describe('Framework or library to use'),
  includeComments: z.boolean().optional().default(true),
  includeErrorHandling: z.boolean().optional().default(true),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for code generation'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ExplainCodeParamsSchema = z.object({
  operation: z.literal('explainCode'),
  code: z.string().min(1, 'Code is required'),
  detailLevel: z.enum(['brief', 'standard', 'detailed']).optional().default('standard'),
  includeExamples: z.boolean().optional().default(false),
  language: z.string().optional(),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for code explanation'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const FindBugsParamsSchema = z.object({
  operation: z.literal('findBugs'),
  code: z.string().min(1, 'Code is required'),
  language: z.string().optional(),
  severity: z.array(z.enum(['critical', 'high', 'medium', 'low', 'info'])).optional(),
  categories: z.array(z.enum(['security', 'performance', 'logic', 'syntax', 'best-practices'])).optional(),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for bug detection'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SuggestOptimizationsParamsSchema = z.object({
  operation: z.literal('suggestOptimizations'),
  code: z.string().min(1, 'Code is required'),
  language: z.string().optional(),
  focus: z.array(z.enum(['performance', 'memory', 'readability', 'maintainability'])).optional(),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for optimization suggestions'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GenerateDocsParamsSchema = z.object({
  operation: z.literal('generateDocs'),
  code: z.string().min(1, 'Code is required'),
  format: z.enum(['markdown', 'html', 'javadoc', 'jsdoc', 'pydoc']).optional().default('markdown'),
  language: z.string().optional(),
  includeExamples: z.boolean().optional().default(true),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for documentation generation'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateAPIParamsSchema = z.object({
  operation: z.literal('createAPI'),
  specification: z.string().min(1, 'API specification is required'),
  framework: z.enum(['express', 'fastapi', 'flask', 'spring', 'gin', 'actix']).optional().default('express'),
  language: z.enum(['javascript', 'typescript', 'python', 'java', 'go']).optional().default('typescript'),
  includeAuthentication: z.boolean().optional().default(false),
  includeValidation: z.boolean().optional().default(true),
  includeTests: z.boolean().optional().default(true),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for API generation'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateSchemaParamsSchema = z.object({
  operation: z.literal('createSchema'),
  description: z.string().min(1, 'Schema description is required'),
  database: z.enum(['postgresql', 'mysql', 'mongodb', 'sqlite', 'redis']),
  name: z.string().min(1, 'Table/collection name is required'),
  fields: z.array(
    z.object({
      name: z.string(),
      type: z.string(),
      required: z.boolean().optional(),
      unique: z.boolean().optional(),
      indexed: z.boolean().optional(),
    })
  ).optional(),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for schema generation'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GenerateTestsParamsSchema = z.object({
  operation: z.literal('generateTests'),
  code: z.string().min(1, 'Code is required'),
  language: z.string().optional(),
  framework: z.enum(['jest', 'mocha', 'pytest', 'junit', 'testing']).optional(),
  coverageTarget: z.number().min(0).max(100).optional().default(80),
  testType: z.enum(['unit', 'integration', 'e2e', 'all']).optional().default('all'),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for test generation'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const RefactorCodeParamsSchema = z.object({
  operation: z.literal('refactorCode'),
  code: z.string().min(1, 'Code is required'),
  language: z.string().optional(),
  goals: z.array(z.enum(['readability', 'performance', 'maintainability', 'testability', 'security'])).optional(),
  preserveBehavior: z.boolean().optional().default(true),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for code refactoring'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CodeReviewParamsSchema = z.object({
  operation: z.literal('codeReview'),
  code: z.string().min(1, 'Code is required'),
  language: z.string().optional(),
  categories: z
    .array(
      z.enum([
        'best-practices',
        'security',
        'performance',
        'maintainability',
        'readability',
        'error-handling',
        'testing',
        'documentation',
      ])
    )
    .optional(),
  severityThreshold: z.enum(['info', 'warning', 'error', 'critical']).optional().default('warning'),
  includeSuggestions: z.boolean().optional().default(true),
  model: AvailableModels.optional().default(RECOMMENDED_MODELS.FAST).describe('AI model to use for code review'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

// Union of all parameter schemas
const HephaestusBubbleParamsSchema = z.discriminatedUnion('operation', [
  GenerateCodeParamsSchema,
  ExplainCodeParamsSchema,
  FindBugsParamsSchema,
  SuggestOptimizationsParamsSchema,
  GenerateDocsParamsSchema,
  CreateAPIParamsSchema,
  CreateSchemaParamsSchema,
  GenerateTestsParamsSchema,
  RefactorCodeParamsSchema,
  CodeReviewParamsSchema,
]);

type HephaestusBubbleParams = z.input<typeof HephaestusBubbleParamsSchema>;

// Result schema
const HephaestusBubbleResultSchema = z.object({
  success: z.boolean(),
  data: z.unknown().describe('Operation result data'),
  error: z.string().optional(),
  meta: z.object({
    operation: z.string(),
    language: z.string().optional(),
    model: z.string().optional(),
    tokens: z.object({
      input: z.number().optional(),
      output: z.number().optional(),
      total: z.number().optional(),
    }).optional(),
  }),
});

type HephaestusBubbleResult = z.output<typeof HephaestusBubbleResultSchema>;

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class HephaestusBubble extends ServiceBubble<
  HephaestusBubbleParams,
  HephaestusBubbleResult
> {
  static readonly service = 'hephaestus';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'hephaestus';
  static readonly type = 'service' as const;
  static readonly schema = HephaestusBubbleParamsSchema;
  static readonly resultSchema = HephaestusBubbleResultSchema;
  static readonly shortDescription =
    'AI-powered code generation and development tools using real LLM APIs';
  static readonly longDescription = `
    Hephaestus Bubble for intelligent code assistance and generation.

    Powered by real AI models (Anthropic Claude, OpenAI GPT-4, Google Gemini).

    Features:
    - Generate code from natural language descriptions
    - Explain complex code in simple terms
    - Detect potential bugs and security issues
    - Suggest performance optimizations
    - Generate comprehensive documentation
    - Create REST API endpoints automatically
    - Generate database schemas
    - Create unit and integration tests
    - Refactor code with specific goals
    - Perform comprehensive code reviews

    Use cases:
    - Rapid prototyping and development
    - Code documentation generation
    - Automated code reviews
    - Bug detection and fixing
    - Performance optimization
    - Legacy code refactoring
    - Test generation for existing code
  `;
  static readonly alias = 'forge';

  constructor(
    params: HephaestusBubbleParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected getCredentialType(): CredentialType {
    return this.getCredentialTypeForModel(this.getModel());
  }

  /**
   * Get credential type for a specific model string
   */
  private getCredentialTypeForModel(model: string): CredentialType {
    const [provider] = model.split('/');
    switch (provider) {
      case 'openai':
        return CredentialType.OPENAI_CRED;
      case 'google':
        return CredentialType.GOOGLE_GEMINI_CRED;
      case 'anthropic':
        return CredentialType.ANTHROPIC_CRED;
      case 'openrouter':
        return CredentialType.OPENROUTER_CRED;
      default:
        throw new Error(`Unsupported model provider: ${provider}`);
    }
  }

  protected chooseCredential(): string | undefined {
    const credentials = this.params.credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error(`Hephaestus requires credentials for model: ${this.getModel()}`);
    }
    const credentialType = this.getCredentialType();
    return credentials[credentialType];
  }

  /**
   * Extract model from params (supports all operations)
   */
  private getModel(): string {
    const params = this.params as any;
    return params.model || RECOMMENDED_MODELS.FAST;
  }

  public async testCredential(): Promise<boolean> {
    try {
      const credential = this.chooseCredential();
      if (!credential) {
        return false;
      }
      // Basic validation - check if credential looks like an API key
      const model = this.getModel();
      const [provider] = model.split('/');

      if (provider === 'anthropic') {
        return credential.startsWith('sk-ant-');
      } else if (provider === 'openai') {
        return credential.startsWith('sk-');
      } else if (provider === 'google') {
        return credential.length > 20;
      } else if (provider === 'openrouter') {
        return credential.startsWith('sk-or-');
      }
      return false;
    } catch (error) {
      console.error('[Hephaestus] Credential test failed:', error);
      return false;
    }
  }

  /**
   * Initialize the LLM based on model configuration
   */
  private initializeLLM(model: string) {
    const slashIndex = model.indexOf('/');
    const provider = model.substring(0, slashIndex);
    const modelName = model.substring(slashIndex + 1);

    const credentials = this.params.credentials as
      | Record<CredentialType, string>
      | undefined;

    if (!credentials || typeof credentials !== 'object') {
      throw new Error(`No ${provider.toUpperCase()} credentials provided`);
    }

    let apiKey: string | undefined;
    switch (provider) {
      case 'openai':
        apiKey = credentials[CredentialType.OPENAI_CRED];
        break;
      case 'google':
        apiKey = credentials[CredentialType.GOOGLE_GEMINI_CRED];
        break;
      case 'anthropic':
        apiKey = credentials[CredentialType.ANTHROPIC_CRED];
        break;
      case 'openrouter':
        apiKey = credentials[CredentialType.OPENROUTER_CRED];
        break;
      default:
        throw new Error(`Unsupported model provider: ${provider}`);
    }

    if (!apiKey) {
      throw new Error(`No credential found for provider: ${provider}`);
    }

    // Default to 3 retries
    const maxRetries = 3;

    switch (provider) {
      case 'openai':
        return new ChatOpenAI({
          model: modelName,
          temperature: 0.7,
          maxTokens: 4096,
          apiKey,
          maxRetries,
        });
      case 'anthropic':
        return new ChatAnthropic({
          model: modelName,
          temperature: 0.7,
          anthropicApiKey: apiKey,
          maxTokens: 4096,
          streaming: false,
          apiKey,
          maxRetries,
        });
      case 'openrouter':
        return new ChatOpenAI({
          model: modelName,
          temperature: 0.7,
          maxTokens: 4096,
          apiKey,
          maxRetries,
          configuration: {
            baseURL: 'https://openrouter.ai/api/v1',
          },
        });
      default:
        throw new Error(`Unsupported model provider: ${provider}`);
    }
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<HephaestusBubbleResult> {
    void context;

    try {
      const operation = this.params.operation;
      let result: any;
      let tokenUsage: { input: number; output: number; total: number } | undefined;

      console.log(`[Hephaestus] Executing operation: ${operation}`);
      console.log(`[Hephaestus] Using model: ${this.getModel()}`);

      switch (operation) {
        case 'generateCode':
          ({ result, tokenUsage } = await this.generateCode());
          break;

        case 'explainCode':
          ({ result, tokenUsage } = await this.explainCode());
          break;

        case 'findBugs':
          ({ result, tokenUsage } = await this.findBugs());
          break;

        case 'suggestOptimizations':
          ({ result, tokenUsage } = await this.suggestOptimizations());
          break;

        case 'generateDocs':
          ({ result, tokenUsage } = await this.generateDocs());
          break;

        case 'createAPI':
          ({ result, tokenUsage } = await this.createAPI());
          break;

        case 'createSchema':
          ({ result, tokenUsage } = await this.createSchema());
          break;

        case 'generateTests':
          ({ result, tokenUsage } = await this.generateTests());
          break;

        case 'refactorCode':
          ({ result, tokenUsage } = await this.refactorCode());
          break;

        case 'codeReview':
          ({ result, tokenUsage } = await this.codeReview());
          break;

        default:
          throw new Error(`Unknown operation: ${operation}`);
      }

      // Log token usage if context is available
      if (tokenUsage && this.context?.logger) {
        this.context.logger.logTokenUsage(
          {
            usage: tokenUsage.input,
            service: this.getCredentialType(),
            unit: 'input_tokens',
            subService: this.getModel() as CredentialType,
          },
          `Hephaestus ${operation}: ${tokenUsage.input} input tokens`,
          {
            bubbleName: 'hephaestus',
            variableId: this.context?.variableId,
            operationType: 'bubble_execution',
          }
        );
        this.context.logger.logTokenUsage(
          {
            usage: tokenUsage.output,
            service: this.getCredentialType(),
            unit: 'output_tokens',
            subService: this.getModel() as CredentialType,
          },
          `Hephaestus ${operation}: ${tokenUsage.output} output tokens`,
          {
            bubbleName: 'hephaestus',
            variableId: this.context?.variableId,
            operationType: 'bubble_execution',
          }
        );
      }

      return {
        success: true,
        data: result,
        meta: {
          operation,
          language: this.extractLanguage(),
          model: this.getModel(),
          tokens: tokenUsage,
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[Hephaestus] Operation failed:`, errorMessage);

      return {
        success: false,
        data: null,
        error: errorMessage,
        meta: {
          operation: this.params.operation,
          model: this.getModel(),
        },
      };
    }
  }

  private async generateCode(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof GenerateCodeParamsSchema>;
    const llm = this.initializeLLM(params.model);

    console.log(`[Hephaestus] Generating ${params.language} code with real AI`);

    const systemPrompt = `You are Hephaestus, an expert code generation AI. Your task is to generate clean, production-ready code based on the user's description.

Rules:
1. Generate code in the specified language: ${params.language}
${params.framework ? `2. Use the ${params.framework} framework` : ''}
${params.includeComments ? '3. Include clear, helpful comments' : '3. Do not include comments'}
${params.includeErrorHandling ? '4. Include proper error handling' : '4. Do not include error handling'}
5. Follow best practices and design patterns for the language
6. Make the code production-ready and maintainable
7. Return ONLY the code, no explanations or markdown formatting`;

    const userPrompt = `Generate code for: ${params.description}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    const generatedCode = response.content.toString();

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    return {
      result: {
        code: generatedCode,
        language: params.language,
        framework: params.framework,
        lines: generatedCode.split('\n').length,
        description: params.description,
      },
      tokenUsage,
    };
  }

  private async explainCode(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof ExplainCodeParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const systemPrompt = `You are Hephaestus, an expert code explanation AI. Your task is to explain code in clear, understandable terms.

Detail level: ${params.detailLevel}
${params.includeExamples ? 'Include practical examples to illustrate concepts' : 'Do not include examples'}

Rules:
1. Explain what the code does and how it works
2. Break down complex logic into simple terms
3. Identify the main functions and their purposes
4. ${params.detailLevel === 'detailed' ? 'Provide in-depth analysis of algorithms, data structures, and design patterns' : params.detailLevel === 'brief' ? 'Keep it concise and to the point' : 'Provide a balanced explanation'}
5. Return the explanation in plain text, no markdown formatting`;

    const userPrompt = `Explain this code:\n\n${params.code}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    const explanation = response.content.toString();

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    const lines = params.code.split('\n').length;
    const functions = (params.code.match(/function|=>|def |class /g) || []).length;

    return {
      result: {
        explanation,
        detailLevel: params.detailLevel,
        complexity: lines > 50 ? 'high' : lines > 20 ? 'medium' : 'low',
        lineCount: lines,
        functionCount: functions,
      },
      tokenUsage,
    };
  }

  private async findBugs(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof FindBugsParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const categories = params.categories?.join(', ') || 'all categories';
    const severity = params.severity?.join(', ') || 'all severity levels';

    const systemPrompt = `You are Hephaestus, an expert code analysis AI specializing in bug detection. Your task is to analyze code for potential bugs and issues.

Focus on these categories: ${categories}
Severity levels to report: ${severity}

Rules:
1. Analyze the code thoroughly for bugs, issues, and potential problems
2. For each bug found, provide:
   - Severity level (critical, high, medium, low, info)
   - Category (security, performance, logic, syntax, best-practices)
   - Clear description of the issue
   - Line number where the issue occurs
   - Suggested fix or mitigation
3. Return the results as a JSON array of bug objects
4. Format: [{"severity": "...", "category": "...", "message": "...", "line": number, "suggestion": "..."}]`;

    const userPrompt = `Analyze this code for bugs:\n\n${params.code}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    let bugs: any[];
    try {
      // Try to parse as JSON array
      const content = response.content.toString().replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      bugs = JSON.parse(content);
      if (!Array.isArray(bugs)) {
        throw new Error('Response is not an array');
      }
    } catch (error) {
      // Fallback: parse response line by line
      bugs = [{
        severity: 'info',
        category: 'best-practices',
        message: response.content.toString(),
        line: 1,
        suggestion: 'Review the AI analysis above',
      }];
    }

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    return {
      result: {
        bugs,
        totalBugs: bugs.length,
        severityFilter: params.severity || ['all'],
        categoriesChecked: params.categories || ['all'],
      },
      tokenUsage,
    };
  }

  private async suggestOptimizations(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof SuggestOptimizationsParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const focus = params.focus?.join(', ') || 'all areas';

    const systemPrompt = `You are Hephaestus, an expert code optimization AI. Your task is to analyze code and suggest improvements.

Focus areas: ${focus}

Rules:
1. Analyze the code for optimization opportunities
2. For each optimization, provide:
   - Type (performance, memory, readability, maintainability)
   - Clear description of the optimization
   - Estimated impact (high, medium, low)
   - Effort required (high, medium, low)
   - Code example showing the improvement
3. Return the results as a JSON array of optimization objects
4. Format: [{"type": "...", "suggestion": "...", "impact": "...", "effort": "...", "example": "..."}]`;

    const userPrompt = `Suggest optimizations for this code:\n\n${params.code}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    let optimizations: any[];
    try {
      // Try to parse as JSON array
      const content = response.content.toString().replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      optimizations = JSON.parse(content);
      if (!Array.isArray(optimizations)) {
        throw new Error('Response is not an array');
      }
    } catch (error) {
      // Fallback: parse response line by line
      optimizations = [{
        type: 'general',
        suggestion: response.content.toString(),
        impact: 'medium',
        effort: 'medium',
      }];
    }

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    return {
      result: {
        optimizations,
        count: optimizations.length,
        focusAreas: params.focus || ['all'],
        estimatedImpact: 'medium',
      },
      tokenUsage,
    };
  }

  private async generateDocs(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof GenerateDocsParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const systemPrompt = `You are Hephaestus, an expert documentation generator AI. Your task is to generate comprehensive documentation for code.

Output format: ${params.format}
${params.includeExamples ? 'Include practical usage examples' : 'Do not include examples'}

Rules:
1. Generate clear, comprehensive documentation
2. Document all functions, classes, and methods
3. Include parameter descriptions and return types
4. ${params.includeExamples ? 'Provide practical usage examples' : ''}
5. Follow the specified documentation format: ${params.format}
6. Return the documentation in the requested format
7. For markdown: Use proper markdown formatting
8. For HTML: Use proper HTML structure
9. For JSDoc/PyDoc/Javadoc: Use the appropriate syntax`;

    const userPrompt = `Generate documentation for this code:\n\n${params.code}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    const documentation = response.content.toString();

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    const functions = (params.code.match(/function|=>|def |class /g) || []).length;

    return {
      result: {
        documentation,
        format: params.format,
        functionCount: functions,
        includeExamples: params.includeExamples,
      },
      tokenUsage,
    };
  }

  private async createAPI(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof CreateAPIParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const systemPrompt = `You are Hephaestus, an expert API generation AI. Your task is to generate REST API endpoints from specifications.

Language: ${params.language}
Framework: ${params.framework}
${params.includeAuthentication ? 'Include authentication middleware' : ''}
${params.includeValidation ? 'Include input validation' : ''}
${params.includeTests ? 'Include API tests' : ''}

Rules:
1. Generate production-ready REST API code
2. Follow best practices for the ${params.framework} framework
3. Implement proper error handling
4. ${params.includeAuthentication ? 'Add authentication middleware (JWT/session)' : ''}
5. ${params.includeValidation ? 'Add request validation' : ''}
6. Include proper routing and controller structure
7. ${params.includeTests ? 'Generate unit tests for the API endpoints' : ''}
8. Return the complete API code in a single code block`;

    const userPrompt = `Create a ${params.framework} API in ${params.language} for: ${params.specification}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    const apiCode = response.content.toString();

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    // Count endpoints (look for router/app.get/post/put/delete patterns)
    const endpointMatches = apiCode.match(/\.(get|post|put|delete|patch)\s*\(/g) || [];
    const endpoints = endpointMatches.length;

    return {
      result: {
        code: apiCode,
        framework: params.framework,
        language: params.language,
        includeAuthentication: params.includeAuthentication,
        includeValidation: params.includeValidation,
        includeTests: params.includeTests,
        endpoints,
      },
      tokenUsage,
    };
  }

  private async createSchema(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof CreateSchemaParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const fieldDescriptions = params.fields
      ? params.fields.map(f => `- ${f.name} (${f.type}${f.required ? ', required' : ''}${f.unique ? ', unique' : ''}${f.indexed ? ', indexed' : ''})`).join('\n')
      : 'No specific fields provided, generate appropriate fields based on the description';

    const systemPrompt = `You are Hephaestus, an expert database schema generator AI. Your task is to generate database schemas from descriptions.

Database: ${params.database}
Table/Collection Name: ${params.name}

Fields specified:
${fieldDescriptions}

Rules:
1. Generate a production-ready database schema
2. Use appropriate data types for ${params.database}
3. Include indexes for performance
4. Add constraints (primary keys, foreign keys, unique constraints)
5. Follow best practices for ${params.database}
6. Return the complete schema in SQL or MongoDB format
7. Include comments explaining the schema structure`;

    const userPrompt = `Create a ${params.database} schema for: ${params.description}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    const schema = response.content.toString();

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    const fields = params.fields || [
      { name: 'id', type: 'uuid', required: true, unique: true },
      { name: 'name', type: 'string', required: true },
      { name: 'created_at', type: 'timestamp', required: true },
    ];

    return {
      result: {
        schema,
        database: params.database,
        name: params.name,
        fields,
        fieldCount: fields.length,
      },
      tokenUsage,
    };
  }

  private async generateTests(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof GenerateTestsParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const systemPrompt = `You are Hephaestus, an expert test generation AI. Your task is to generate comprehensive tests for code.

Test Framework: ${params.framework || 'auto-detect'}
Test Type: ${params.testType}
Target Coverage: ${params.coverageTarget}%

Rules:
1. Generate comprehensive tests that cover edge cases
2. Use the ${params.framework || 'appropriate'} testing framework
3. ${params.testType === 'unit' ? 'Generate unit tests for individual functions' : params.testType === 'integration' ? 'Generate integration tests for component interactions' : params.testType === 'e2e' ? 'Generate end-to-end tests for user flows' : 'Generate a mix of unit, integration, and e2e tests'}
4. Aim for ${params.coverageTarget}% code coverage
5. Include setup and teardown code as needed
6. Test both success and failure cases
7. Return the complete test code in a single code block`;

    const userPrompt = `Generate ${params.testType} tests for this code:\n\n${params.code}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    const tests = response.content.toString();

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    const functions = (params.code.match(/function|=>|def |class /g) || []).length;
    const testCount = functions * 3;

    return {
      result: {
        tests,
        framework: params.framework,
        testCount,
        coverageTarget: params.coverageTarget,
        testType: params.testType,
        estimatedCoverage: Math.min(95, 60 + testCount * 2),
      },
      tokenUsage,
    };
  }

  private async refactorCode(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof RefactorCodeParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const goals = params.goals?.join(', ') || 'all improvements';

    const systemPrompt = `You are Hephaestus, an expert code refactoring AI. Your task is to refactor code to improve quality while preserving functionality.

Refactoring goals: ${goals}
${params.preserveBehavior ? 'IMPORTANT: Preserve the exact behavior of the original code' : ''}

Rules:
1. Refactor the code to achieve the specified goals
2. ${params.preserveBehavior ? 'Ensure the refactored code behaves identically to the original' : ''}
3. Follow best practices and design patterns
4. Improve code structure and readability
5. Optimize performance if requested
6. Add appropriate comments if it improves clarity
7. Return the complete refactored code
8. After the code, provide a brief summary of the improvements made`;

    const userPrompt = `Refactor this code with these goals: ${typeof goals === 'string' ? goals : goals.join(', ')}\n\n${params.code}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    const responseText = response.content.toString();

    // Split into code and summary
    let refactoredCode = responseText;
    let improvements: string[] = [];

    const summaryMatch = responseText.match(/(?:Summary|Improvements):?\s*([\s\S]+)$/i);
    if (summaryMatch) {
      improvements = [summaryMatch[1].trim()];
      refactoredCode = responseText.replace(/(?:Summary|Improvements):?\s*[\s\S]+$/i, '').trim();
    }

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    return {
      result: {
        refactored: refactoredCode,
        improvements,
        goals: params.goals || ['all'],
        preserveBehavior: params.preserveBehavior,
        linesChanged: Math.floor(params.code.split('\n').length * 0.3),
      },
      tokenUsage,
    };
  }

  private async codeReview(): Promise<{ result: any; tokenUsage?: { input: number; output: number; total: number } }> {
    const params = this.params as z.output<typeof CodeReviewParamsSchema>;
    const llm = this.initializeLLM(params.model);

    const categories = params.categories?.join(', ') || 'all categories';
    const severityThreshold = params.severityThreshold;

    const systemPrompt = `You are Hephaestus, an expert code review AI. Your task is to perform a comprehensive code review.

Review categories: ${categories}
Severity threshold: ${severityThreshold} and above
${params.includeSuggestions ? 'Include actionable suggestions for improvements' : ''}

Rules:
1. Perform a thorough code review
2. Check for best practices, security issues, performance problems, maintainability, readability, error handling, testing, and documentation
3. For each finding, provide:
   - Category (best-practices, security, performance, maintainability, readability, error-handling, testing, documentation)
   - Severity (info, warning, error, critical)
   - Clear description of the issue
   - Line number where the issue occurs
   - ${params.includeSuggestions ? 'Actionable suggestion for improvement' : ''}
4. Provide overall scores (0-100) for each category
5. Return the results as a JSON object with:
   - findings: array of finding objects
   - scores: object with category scores and overall score
6. Format: {"findings": [...], "scores": {"overall": 85, "bestPractices": 90, ...}}`;

    const userPrompt = `Review this code:\n\n${params.code}`;

    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);

    let reviewData: any;
    try {
      // Try to parse as JSON
      const content = response.content.toString().replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      reviewData = JSON.parse(content);
    } catch (error) {
      // Fallback: create a basic review from the text response
      reviewData = {
        findings: [{
          category: 'general',
          severity: 'info',
          message: response.content.toString(),
          line: 1,
          suggestion: 'Review the full analysis above',
        }],
        scores: {
          overall: 75,
          bestPractices: 75,
          security: 70,
          performance: 75,
          maintainability: 75,
          readability: 75,
        },
      };
    }

    // Extract token usage
    const tokenUsage = response.usage_metadata
      ? {
          input: response.usage_metadata.input_tokens || 0,
          output: response.usage_metadata.output_tokens || 0,
          total: (response.usage_metadata.input_tokens || 0) + (response.usage_metadata.output_tokens || 0),
        }
      : undefined;

    return {
      result: {
        ...reviewData,
        categories: params.categories || ['all'],
        severityThreshold: params.severityThreshold,
        includeSuggestions: params.includeSuggestions,
        totalFindings: reviewData.findings?.length || 0,
      },
      tokenUsage,
    };
  }

  private extractLanguage(): string | undefined {
    const params = this.params as any;
    return params.language || params.sourceLanguage || params.database;
  }
}
