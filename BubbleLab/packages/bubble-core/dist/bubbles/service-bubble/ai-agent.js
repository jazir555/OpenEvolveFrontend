import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType, BUBBLE_CREDENTIAL_OPTIONS, RECOMMENDED_MODELS, } from '@bubblelab/shared-schemas';
import { StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { HumanMessage, AIMessage, ToolMessage, AIMessageChunk, } from '@langchain/core/messages';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AvailableModels } from '@bubblelab/shared-schemas';
import { AvailableTools, } from '../../types/available-tools.js';
import { BubbleFactory } from '../../bubble-factory.js';
import { ConversationMessageSchema } from '@bubblelab/shared-schemas';
import { extractAndStreamThinkingTokens, formatFinalResponse, generationsToMessageContent, } from '../../utils/agent-formatter.js';
import { isAIMessage, isAIMessageChunk } from '@langchain/core/messages';
import { HarmBlockThreshold, HarmCategory } from '@google/generative-ai';
import { SafeGeminiChat } from '../../utils/safe-gemini-chat.js';
import { zodSchemaToJsonString, buildJsonSchemaInstruction, } from '../../utils/zod-schema.js';
import { Logger } from '../../utils/logger.js';
import { AI_AGENT } from '../../utils/constants.js';
// Define backup model configuration schema
const BackupModelConfigSchema = z.object({
    model: AvailableModels.describe('Backup AI model to use if the primary model fails (format: provider/model-name).'),
    temperature: z
        .number()
        .min(0)
        .max(2)
        .optional()
        .describe('Temperature for backup model. If not specified, uses primary model temperature.'),
    maxTokens: z
        .number()
        .positive()
        .optional()
        .describe('Max tokens for backup model. If not specified, uses primary model maxTokens.'),
    reasoningEffort: z
        .enum(['low', 'medium', 'high'])
        .optional()
        .describe('Reasoning effort for backup model. If not specified, uses primary model reasoningEffort.'),
    maxRetries: z
        .number()
        .int()
        .min(0)
        .max(10)
        .optional()
        .describe('Max retries for backup model. If not specified, uses primary model maxRetries.'),
});
// Define model configuration
const ModelConfigSchema = z.object({
    model: AvailableModels.describe('AI model to use (format: provider/model-name).'),
    temperature: z
        .number()
        .min(0)
        .max(2)
        .default(1)
        .describe('Temperature for response randomness (0 = deterministic, 2 = very random)'),
    maxTokens: z
        .number()
        .positive()
        .optional()
        .default(64000)
        .describe('Maximum number of tokens to generate in response, keep at default of 40000 unless the response is expected to be certain length'),
    reasoningEffort: z
        .enum(['low', 'medium', 'high'])
        .optional()
        .describe('Reasoning effort for model. If not specified, uses primary model reasoningEffort.'),
    maxRetries: z
        .number()
        .int()
        .min(0)
        .max(10)
        .default(3)
        .describe('Maximum number of retries for API calls (default: 3). Useful for handling transient errors like 503 Service Unavailable.'),
    provider: z
        .array(z.string())
        .optional()
        .describe('Providers for ai agent (open router only).'),
    jsonMode: z
        .boolean()
        .default(false)
        .describe('When true, returns clean JSON response, you must provide the exact JSON schema in the system prompt'),
    backupModel: BackupModelConfigSchema.default({
        model: RECOMMENDED_MODELS.FAST,
    })
        .optional()
        .describe('Backup model configuration to use if the primary model fails.'),
});
// Define tool configuration for pre-registered tools
const ToolConfigSchema = z.object({
    name: AvailableTools.describe('Name of the tool type or tool bubble to enable for the AI agent'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .default({})
        .optional()
        .describe('Credential types to use for the tool bubble (injected at runtime)'),
    config: z
        .record(z.string(), z.unknown())
        .optional()
        .describe('Configuration for the tool or tool bubble'),
});
// SECURITY FIX: Disable customTools feature entirely to prevent arbitrary code execution
// This feature allows users to execute arbitrary JavaScript, which is a critical security risk
// Use pre-registered tools from the factory instead, which are audited and safe
// Define custom tool schema for runtime-defined tools
const CustomToolSchema = z
    .object({
    name: z
        .string()
        .min(1)
        .describe('Unique name for your custom tool (e.g., "calculate-tax")'),
    description: z
        .string()
        .min(1)
        .describe('Description of what the tool does - helps the AI know when to use it'),
    schema: z
        .union([
        z.record(z.string(), z.unknown()),
        z.custom((val) => val && typeof val === 'object' && '_def' in val),
    ])
        .describe('Zod schema object defining the tool parameters. Can be either a plain object (e.g., { amount: z.number() }) or a Zod object directly (e.g., z.object({ amount: z.number() })).'),
    func: z
        .function()
        .args(z.record(z.string(), z.unknown()))
        .returns(z.promise(z.unknown()))
        .describe('⛔ SECURITY RISK: Custom tools are DISABLED for security reasons. Use pre-registered tools from the factory instead.'),
})
    .refine(() => false, '⛔ SECURITY: Custom tools are disabled to prevent arbitrary code execution. Use the pre-registered tools from the factory (tools parameter) instead.');
// Define image input schemas - supports both base64 data and URLs
const Base64ImageSchema = z.object({
    type: z.literal('base64').default('base64'),
    data: z
        .string()
        .describe('Base64 encoded image data (without data:image/... prefix)'),
    mimeType: z
        .string()
        .default('image/png')
        .describe('MIME type of the image (e.g., image/png, image/jpeg)'),
    description: z
        .string()
        .optional()
        .describe('Optional description or context for the image'),
});
const UrlImageSchema = z.object({
    type: z.literal('url'),
    url: z
        .string()
        .url()
        .describe('URL to the image (http/https)')
        .refine((url) => {
        // SECURITY FIX: Validate URL to prevent SSRF attacks
        try {
            const parsedUrl = new URL(url);
            // Only allow http and https protocols
            if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
                return false;
            }
            // Block private/internal IP ranges to prevent SSRF
            const hostname = parsedUrl.hostname.toLowerCase();
            // Block localhost variants
            if (hostname === 'localhost' ||
                hostname === '127.0.0.1' ||
                hostname.startsWith('127.') ||
                hostname === '[::1]' ||
                hostname === '0.0.0.0') {
                return false;
            }
            // Block private IP ranges (CIDR notation)
            // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            const privateIpPatterns = [
                /^10\./,
                /^172\.(1[6-9]|2\d|3[01])\./,
                /^192\.168\./,
                /^169\.254\./, // Link-local
            ];
            if (privateIpPatterns.some((pattern) => pattern.test(hostname))) {
                return false;
            }
            // Block internal hostnames
            const internalHostnames = [
                'metadata.google.internal',
                'instance-data',
                'linklocal.amazonaws.com',
            ];
            if (internalHostnames.includes(hostname)) {
                return false;
            }
            // Block file:// protocol (already caught by protocol check)
            // Block cloud metadata endpoints
            const metadataEndpoints = [
                'metadata',
                '169.254.169.254',
                '100.100.100.200', // GCP metadata
            ];
            if (metadataEndpoints.some((endpoint) => hostname.includes(endpoint))) {
                return false;
            }
            return true;
        }
        catch {
            return false;
        }
    }, 'URL contains forbidden protocol, internal IP address, or private range'),
    description: z
        .string()
        .optional()
        .describe('Optional description or context for the image'),
});
const ImageInputSchema = z.discriminatedUnion('type', [
    Base64ImageSchema,
    UrlImageSchema,
]);
// Schema for the expected JSON output structure - accepts either a Zod schema or a JSON schema string
const ExpectedOutputSchema = z.union([
    z.custom((val) => val?._def !== undefined),
    z.string(),
]);
// Define the parameters schema for the AI Agent bubble
const AIAgentParamsSchema = z.object({
    message: z
        .string()
        .min(1, 'Message is required')
        .describe('The message or question to send to the AI agent'),
    images: z
        .array(ImageInputSchema)
        .default([])
        .describe('Array of base64 encoded images to include with the message (for multimodal AI models). Example: [{type: "base64", data: "base64...", mimeType: "image/png", description: "A beautiful image of a cat"}] or [{type: "url", url: "https://example.com/image.png", description: "A beautiful image of a cat"}]'),
    conversationHistory: z
        .array(ConversationMessageSchema)
        .optional()
        .describe('Previous conversation messages for multi-turn conversations. When provided, messages are sent as separate turns to enable KV cache optimization. Format: [{role: "user", content: "..."}, {role: "assistant", content: "..."}, ...]'),
    systemPrompt: z
        .string()
        .default('You are a helpful AI assistant')
        .describe('System prompt that defines the AI agents behavior and personality'),
    name: z
        .string()
        .default('AI Agent')
        .optional()
        .describe('A friendly name for the AI agent'),
    model: ModelConfigSchema.default({
        model: RECOMMENDED_MODELS.FAST,
        temperature: 1,
        maxTokens: 65536,
        maxRetries: 3,
        jsonMode: false,
    }).describe('AI model configuration including provider, temperature, and tokens, retries, and json mode. Always include this.'),
    tools: z
        .array(ToolConfigSchema)
        .default([])
        .describe('Array of pre-registered tools the AI agent can use. Can be tool types (web-search-tool, web-scrape-tool, web-crawl-tool, web-extract-tool, instagram-tool). If using image models, set the tools to []'),
    customTools: z
        .array(CustomToolSchema)
        .default([])
        .optional()
        .describe('Array of custom runtime-defined tools with their own schemas and functions. Use this to add domain-specific tools without pre-registration. Example: [{ name: "calculate-tax", description: "Calculates sales tax", schema: { amount: z.number() }, func: async (input) => {...} }]'),
    maxIterations: z
        .number()
        .int()
        .positive()
        .min(5, 'maxIterations must be at least 5 to support multi-step reasoning')
        .default(40)
        .describe('Maximum number of iterations for the agent workflow, 5 iterations per turn of conversation'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Object mapping credential types to values (injected at runtime)'),
    streaming: z
        .boolean()
        .default(false)
        .describe('Enable real-time streaming of tokens, tool calls, and iteration progress'),
    expectedOutputSchema: ExpectedOutputSchema.optional().describe('Zod schema or JSON schema string that defines the expected structure of the AI response. When provided, automatically enables JSON mode and instructs the AI to output in the exact format. Example: z.object({ summary: z.string(), items: z.array(z.object({ name: z.string(), score: z.number() })) })'),
    // Note: beforeToolCall and afterToolCall are function hooks added via TypeScript interface
    // They cannot be part of the Zod schema but are available in the params
});
const AIAgentResultSchema = z.object({
    response: z
        .string()
        .describe('The AI agents final response to the user message. For text responses, returns plain text. If JSON mode is enabled, returns a JSON string. For image generation models (like gemini-2.5-flash-image-preview), returns base64-encoded image data with data URI format (data:image/png;base64,...)'),
    toolCalls: z
        .array(z.object({
        tool: z.string().describe('Name of the tool that was called'),
        input: z.unknown().describe('Input parameters passed to the tool'),
        output: z.unknown().describe('Output returned by the tool'),
    }))
        .describe('Array of tool calls made during the conversation'),
    iterations: z
        .number()
        .describe('Number of back-and-forth iterations in the agent workflow'),
    error: z
        .string()
        .describe('Error message of the run, undefined if successful'),
    success: z
        .boolean()
        .describe('Whether the agent execution completed successfully'),
});
/**
 * LRU Cache for conversation history and tool results
 * Prevents unbounded memory growth with TTL-based cleanup
 */
class LRUCache {
    cache;
    maxSize;
    ttl;
    constructor(maxSize, ttl) {
        this.cache = new Map();
        this.maxSize = maxSize;
        this.ttl = ttl;
    }
    set(key, value) {
        // Remove oldest entry if at capacity
        if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, { value, timestamp: Date.now() });
    }
    get(key) {
        const entry = this.cache.get(key);
        if (!entry)
            return undefined;
        // Check if entry has expired
        if (Date.now() - entry.timestamp > this.ttl) {
            this.cache.delete(key);
            return undefined;
        }
        // Move to end (most recently used)
        this.cache.delete(key);
        this.cache.set(key, entry);
        return entry.value;
    }
    has(key) {
        return this.get(key) !== undefined;
    }
    clear() {
        this.cache.clear();
    }
    get size() {
        return this.cache.size;
    }
    /**
     * Cleanup expired entries
     */
    cleanup() {
        const now = Date.now();
        let removed = 0;
        for (const [key, entry] of this.cache.entries()) {
            if (now - entry.timestamp > this.ttl) {
                this.cache.delete(key);
                removed++;
            }
        }
        return removed;
    }
}
export class AIAgentBubble extends ServiceBubble {
    static type = 'service';
    static service = 'ai-agent';
    static authType = 'apikey';
    static bubbleName = 'ai-agent';
    static schema = AIAgentParamsSchema;
    static resultSchema = AIAgentResultSchema;
    static shortDescription = 'AI agent with LangGraph for tool-enabled conversations, multimodal support, and JSON mode';
    static longDescription = `
    An AI agent powered by LangGraph that can use any tool bubble to answer questions.
    Use cases:
    - Add tools to enhance the AI agent's capabilities (web-search-tool, web-scrape-tool)
    - Multi-step reasoning with tool assistance
    - Tool-augmented conversations with any registered tool
    - JSON mode for structured output (strips markdown formatting)
  `;
    static alias = 'agent';
    // Static LRU caches with size limits and TTL
    static conversationCache = new LRUCache(AI_AGENT.CACHE_SIZE, AI_AGENT.CACHE_TTL_MS);
    static toolResultCache = new LRUCache(AI_AGENT.CACHE_SIZE * AI_AGENT.MAX_RETRIES_DEFAULT, AI_AGENT.CACHE_TTL_SHORT_MS);
    static CLEANUP_INTERVAL = AI_AGENT.CLEANUP_INTERVAL_MS;
    static lastCleanup = Date.now();
    factory;
    beforeToolCallHook;
    afterToolCallHook;
    afterLLMCallHook;
    streamingCallback;
    shouldStopAfterTools = false;
    shouldContinueToAgent = false;
    logger;
    constructor(params = {
        message: 'Hello, how are you?',
        systemPrompt: 'You are a helpful AI assistant',
        model: { model: RECOMMENDED_MODELS.FAST },
    }, context, instanceId) {
        super(params, context, instanceId);
        this.beforeToolCallHook = params.beforeToolCall;
        this.afterToolCallHook = params.afterToolCall;
        this.afterLLMCallHook = params.afterLLMCall;
        this.streamingCallback = params.streamingCallback;
        this.factory = new BubbleFactory();
        this.logger = new Logger('AIAgent');
        // Periodic cleanup of expired cache entries
        this.performCacheCleanup();
    }
    /**
     * Perform periodic cleanup of cache entries
     */
    performCacheCleanup() {
        const now = Date.now();
        if (now - AIAgentBubble.lastCleanup > AIAgentBubble.CLEANUP_INTERVAL) {
            const convRemoved = AIAgentBubble.conversationCache.cleanup();
            const toolRemoved = AIAgentBubble.toolResultCache.cleanup();
            AIAgentBubble.lastCleanup = now;
            if (convRemoved > 0 || toolRemoved > 0) {
                this.logger.info('Cache cleanup completed', {
                    conversations_removed: convRemoved,
                    tool_results_removed: toolRemoved,
                });
            }
        }
    }
    async testCredential() {
        // Make a test API call to the model provider
        const llm = this.initializeModel(this.params.model);
        const response = await llm.invoke(['Hello, how are you?']);
        if (response.content) {
            return true;
        }
        return false;
    }
    /**
     * Build effective model config from primary and optional backup settings
     */
    buildModelConfig(primaryConfig, backupConfig) {
        if (!backupConfig) {
            return primaryConfig;
        }
        return {
            model: backupConfig.model,
            temperature: backupConfig.temperature ?? primaryConfig.temperature,
            maxTokens: backupConfig.maxTokens ?? primaryConfig.maxTokens,
            maxRetries: backupConfig.maxRetries ?? primaryConfig.maxRetries,
            provider: primaryConfig.provider,
            jsonMode: primaryConfig.jsonMode,
            backupModel: undefined, // Don't chain backup models
        };
    }
    /**
     * Core execution logic for running the agent with a given model config
     */
    async executeWithModel(modelConfig) {
        const { message, images, systemPrompt, tools, customTools, maxIterations, conversationHistory, } = this.params;
        // Initialize the language model
        const llm = this.initializeModel(modelConfig);
        // Initialize tools (both pre-registered and custom)
        const agentTools = await this.initializeTools(tools, customTools);
        // Create the agent graph
        const graph = await this.createAgentGraph(llm, agentTools, systemPrompt);
        // Execute the agent
        return this.executeAgent(graph, message, images, maxIterations, modelConfig, conversationHistory);
    }
    /**
     * Modify params before execution - centralizes all param transformations
     */
    beforeAction() {
        // Auto-enable JSON mode when expectedOutputSchema is provided
        if (this.params.expectedOutputSchema) {
            this.params.model.jsonMode = true;
            // Enhance system prompt with JSON schema instructions
            const schemaString = zodSchemaToJsonString(this.params.expectedOutputSchema);
            this.params.systemPrompt = `${this.params.systemPrompt}\n\n${buildJsonSchemaInstruction(schemaString)}`;
        }
    }
    async performAction(context) {
        // Context is available but not currently used in this implementation
        void context;
        // Apply param transformations before execution
        this.beforeAction();
        try {
            return await this.executeWithModel(this.params.model);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            this.logger.error('Execution error', error, { error_message: errorMessage });
            // Return error information but mark as recoverable
            return {
                response: `Error: ${errorMessage}`,
                success: false,
                toolCalls: [],
                error: errorMessage,
                iterations: 0,
            };
        }
    }
    getCredentialType() {
        return this.getCredentialTypeForModel(this.params.model.model);
    }
    /**
     * Get credential type for a specific model string
     */
    getCredentialTypeForModel(model) {
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
            case 'deepseek':
                return CredentialType.DEEPSEEK_CRED;
            default:
                throw new Error(`Unsupported model provider: ${provider}`);
        }
    }
    chooseCredential() {
        const { model } = this.params;
        const credentials = this.params.credentials;
        const [provider] = model.model.split('/');
        // If no credentials were injected, throw error immediately (like PostgreSQL)
        if (!credentials || typeof credentials !== 'object') {
            throw new Error(`No ${provider.toUpperCase()} credentials provided`);
        }
        // Choose credential based on the model provider
        switch (provider) {
            case 'openai':
                return credentials[CredentialType.OPENAI_CRED];
            case 'google':
                return credentials[CredentialType.GOOGLE_GEMINI_CRED];
            case 'anthropic':
                return credentials[CredentialType.ANTHROPIC_CRED];
            case 'openrouter':
                return credentials[CredentialType.OPENROUTER_CRED];
            case 'deepseek':
                return credentials[CredentialType.DEEPSEEK_CRED];
            default:
                throw new Error(`Unsupported model provider: ${provider}`);
        }
    }
    initializeModel(modelConfig) {
        const { model, temperature, maxTokens, maxRetries } = modelConfig;
        const slashIndex = model.indexOf('/');
        const provider = model.substring(0, slashIndex);
        const modelName = model.substring(slashIndex + 1);
        const reasoningEffort = modelConfig.reasoningEffort;
        // Get credential based on the modelConfig's provider (not this.params.model)
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error(`No ${provider.toUpperCase()} credentials provided`);
        }
        let apiKey;
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
            case 'deepseek':
                apiKey = credentials[CredentialType.DEEPSEEK_CRED];
                break;
            default:
                throw new Error(`Unsupported model provider: ${provider}`);
        }
        if (!apiKey) {
            throw new Error(`No credential found for provider: ${provider}`);
        }
        // Enable streaming if streamingCallback is provided
        const enableStreaming = !!this.streamingCallback;
        // Default to 3 retries if not specified
        const retries = maxRetries ?? AI_AGENT.MAX_RETRIES_DEFAULT;
        switch (provider) {
            case 'openai':
                return new ChatOpenAI({
                    model: modelName,
                    temperature,
                    maxTokens,
                    apiKey,
                    ...(reasoningEffort && {
                        reasoning: {
                            effort: reasoningEffort,
                            summary: 'auto',
                        },
                    }),
                    streaming: enableStreaming,
                    maxRetries: retries,
                });
            case 'google': {
                const thinkingConfig = reasoningEffort
                    ? {
                        includeThoughts: reasoningEffort ? true : false,
                        thinkingBudget: reasoningEffort === 'low'
                            ? AI_AGENT.THINKING_BUDGET_LOW
                            : reasoningEffort === 'medium'
                                ? AI_AGENT.THINKING_BUDGET_MEDIUM
                                : AI_AGENT.THINKING_BUDGET_HIGH,
                    }
                    : undefined;
                return new SafeGeminiChat({
                    model: modelName,
                    temperature,
                    maxOutputTokens: maxTokens,
                    ...(thinkingConfig && { thinkingConfig }),
                    apiKey,
                    // 3.0 pro preview does breaks with streaming, disabled temporarily until fixed
                    streaming: false,
                    maxRetries: retries,
                    // Disable all safety filters to prevent candidateContent.parts.reduce errors
                    // when Gemini blocks content and returns candidates without content field
                    safetySettings: [
                        {
                            category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold: HarmBlockThreshold.BLOCK_NONE,
                        },
                        {
                            category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold: HarmBlockThreshold.BLOCK_NONE,
                        },
                        {
                            category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold: HarmBlockThreshold.BLOCK_NONE,
                        },
                        {
                            category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold: HarmBlockThreshold.BLOCK_NONE,
                        },
                    ],
                });
            }
            case 'anthropic': {
                // Configure Anthropic "thinking" only when reasoning is enabled.
                // Anthropic's API does not allow `budget_tokens` when thinking is disabled.
                const thinkingConfig = reasoningEffort != null
                    ? {
                        type: 'enabled',
                        budget_tokens: reasoningEffort === 'low'
                            ? AI_AGENT.THINKING_BUDGET_LOW
                            : reasoningEffort === 'medium'
                                ? AI_AGENT.THINKING_BUDGET_MEDIUM
                                : AI_AGENT.THINKING_BUDGET_HIGH,
                    }
                    : undefined;
                return new ChatAnthropic({
                    model: modelName,
                    temperature,
                    anthropicApiKey: apiKey,
                    maxTokens,
                    streaming: true,
                    apiKey,
                    ...(thinkingConfig && { thinking: thinkingConfig }),
                    maxRetries: retries,
                });
            }
            case 'openrouter':
                this.logger.debug('Using OpenRouter provider', { model: modelName });
                return new ChatOpenAI({
                    model: modelName,
                    __includeRawResponse: true,
                    temperature,
                    maxTokens,
                    apiKey,
                    streaming: enableStreaming,
                    maxRetries: retries,
                    configuration: {
                        baseURL: 'https://openrouter.ai/api/v1',
                    },
                    modelKwargs: {
                        provider: {
                            order: this.params.model.provider,
                        },
                        reasoning: {
                            effort: reasoningEffort ?? 'medium',
                            exclude: false,
                        },
                    },
                });
            case 'deepseek':
                return new ChatOpenAI({
                    model: modelName,
                    temperature,
                    maxTokens: Math.min(maxTokens || AI_AGENT.MAX_TOKENS_MEDIUM, AI_AGENT.MAX_TOKENS_MEDIUM),
                    apiKey,
                    streaming: enableStreaming,
                    maxRetries: retries,
                    configuration: {
                        baseURL: 'https://api.deepseek.com',
                    },
                });
            default:
                throw new Error(`Unsupported model provider: ${provider}`);
        }
    }
    async initializeTools(toolConfigs, customToolConfigs) {
        const tools = [];
        await this.factory.registerDefaults();
        // First, initialize custom tools
        for (const customTool of customToolConfigs ?? []) {
            try {
                this.logger.debug('Initializing custom tool', { tool_name: customTool.name });
                // Handle both plain object and Zod object schemas
                let schema;
                if (customTool.schema &&
                    typeof customTool.schema === 'object' &&
                    '_def' in customTool.schema) {
                    // Already a Zod schema object, use it directly
                    schema = customTool.schema;
                }
                else {
                    // Plain object, convert to Zod object
                    schema = z.object(customTool.schema);
                }
                const dynamicTool = new DynamicStructuredTool({
                    name: customTool.name,
                    description: customTool.description,
                    schema: schema,
                    func: customTool.func,
                });
                tools.push(dynamicTool);
            }
            catch (error) {
                this.logger.error('Custom tool initialization failed', error, {
                    tool_name: customTool.name,
                });
                // Continue with other tools even if one fails
                continue;
            }
        }
        // Then, initialize pre-registered tools from factory
        for (const toolConfig of toolConfigs) {
            try {
                const ToolBubbleClass = this.factory.get(toolConfig.name);
                if (!ToolBubbleClass) {
                    this.logger.warn('Tool bubble not found in factory', {
                        tool_name: toolConfig.name,
                    });
                    continue;
                }
                // Check if it's a tool bubble (has toAgentTool method)
                if (!('type' in ToolBubbleClass) || ToolBubbleClass.type !== 'tool') {
                    this.logger.warn('Bubble is not a tool bubble', {
                        bubble_name: toolConfig.name,
                    });
                    continue;
                }
                // Convert to LangGraph tool and add to tools array
                if (!ToolBubbleClass.toolAgent) {
                    this.logger.warn('Tool bubble does not have toolAgent method', {
                        tool_name: toolConfig.name,
                    });
                    continue;
                }
                // Get tool's credential requirements and pass relevant credentials from AI agent
                const toolCredentialOptions = BUBBLE_CREDENTIAL_OPTIONS[toolConfig.name] || [];
                const toolCredentials = {};
                // Pass AI agent's credentials to tools that need them
                for (const credType of toolCredentialOptions) {
                    if (this.params.credentials && this.params.credentials[credType]) {
                        toolCredentials[credType] = this.params.credentials[credType];
                    }
                }
                // Merge with any explicitly provided tool credentials (explicit ones take precedence)
                const finalToolCredentials = {
                    ...toolCredentials,
                    ...(toolConfig.credentials || {}),
                };
                this.logger.debug('Passing credentials to tool', {
                    tool_name: toolConfig.name,
                    credential_keys: Object.keys(finalToolCredentials),
                });
                const langGraphTool = ToolBubbleClass.toolAgent(finalToolCredentials, toolConfig.config || {}, this.context);
                const dynamicTool = new DynamicStructuredTool({
                    name: langGraphTool.name,
                    description: langGraphTool.description,
                    schema: langGraphTool.schema,
                    func: langGraphTool.func,
                });
                tools.push(dynamicTool);
            }
            catch (error) {
                this.logger.error('Tool initialization failed', error, {
                    tool_name: toolConfig.name,
                });
                // Continue with other tools even if one fails
                continue;
            }
        }
        return tools;
    }
    /**
     * Custom tool execution node that supports hooks
     */
    async executeToolsWithHooks(state, tools) {
        const { messages } = state;
        const lastMessage = messages[messages.length - 1];
        const toolCalls = lastMessage.tool_calls || [];
        const toolMessages = [];
        let currentMessages = [...messages];
        // Reset stop flag at the start of tool execution
        this.shouldStopAfterTools = false;
        // Execute each tool call
        for (const toolCall of toolCalls) {
            const tool = tools.find((t) => t.name === toolCall.name);
            if (!tool) {
                this.logger.warn('Tool not found', { tool_name: toolCall.name });
                const errorContent = `Error: Tool ${toolCall.name} not found`;
                const startTime = Date.now();
                // Send tool_start event
                this.streamingCallback?.({
                    type: 'tool_start',
                    data: {
                        tool: toolCall.name,
                        input: toolCall.args,
                        callId: toolCall.id,
                    },
                });
                // Send tool_complete event with error
                this.streamingCallback?.({
                    type: 'tool_complete',
                    data: {
                        callId: toolCall.id,
                        input: toolCall.args,
                        tool: toolCall.name,
                        output: { error: errorContent },
                        duration: Date.now() - startTime,
                    },
                });
                toolMessages.push(new ToolMessage({
                    content: errorContent,
                    tool_call_id: toolCall.id,
                }));
                continue;
            }
            const startTime = Date.now();
            try {
                // Call beforeToolCall hook if provided
                const hookResult_before = await this.beforeToolCallHook?.({
                    toolName: toolCall.name,
                    toolInput: toolCall.args,
                    messages: currentMessages,
                });
                this.streamingCallback?.({
                    type: 'tool_start',
                    data: {
                        tool: toolCall.name,
                        input: toolCall.args,
                        callId: toolCall.id,
                    },
                });
                // If hook returns modified messages/toolInput, apply them
                if (hookResult_before) {
                    if (hookResult_before.messages) {
                        currentMessages = hookResult_before.messages;
                    }
                    toolCall.args = hookResult_before.toolInput;
                }
                // Execute the tool
                const toolOutput = await tool.invoke(toolCall.args);
                // Create tool message
                const toolMessage = new ToolMessage({
                    content: typeof toolOutput === 'string'
                        ? toolOutput
                        : JSON.stringify(toolOutput),
                    tool_call_id: toolCall.id,
                });
                toolMessages.push(toolMessage);
                currentMessages = [...currentMessages, toolMessage];
                // Call afterToolCall hook if provided
                const hookResult_after = await this.afterToolCallHook?.({
                    toolName: toolCall.name,
                    toolInput: toolCall.args,
                    toolOutput,
                    messages: currentMessages,
                });
                // If hook returns modified messages, update current messages
                if (hookResult_after) {
                    if (hookResult_after.messages) {
                        currentMessages = hookResult_after.messages;
                    }
                    // Check if hook wants to stop execution
                    if (hookResult_after.shouldStop === true) {
                        this.shouldStopAfterTools = true;
                    }
                }
                this.streamingCallback?.({
                    type: 'tool_complete',
                    data: {
                        callId: toolCall.id,
                        input: toolCall.args,
                        tool: toolCall.name,
                        output: toolOutput,
                        duration: Date.now() - startTime,
                    },
                });
            }
            catch (error) {
                this.logger.error('Tool execution failed', error, {
                    tool_name: toolCall.name,
                });
                const errorContent = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
                const errorMessage = new ToolMessage({
                    content: errorContent,
                    tool_call_id: toolCall.id,
                });
                toolMessages.push(errorMessage);
                currentMessages = [...currentMessages, errorMessage];
                // Send tool_complete event even on failure so frontend can track it properly
                this.streamingCallback?.({
                    type: 'tool_complete',
                    data: {
                        callId: toolCall.id,
                        input: toolCall.args,
                        tool: toolCall.name,
                        output: { error: errorContent },
                        duration: Date.now() - startTime,
                    },
                });
            }
        }
        // Return the updated messages
        // If hooks modified messages, use those; otherwise use the original messages + tool messages
        if (currentMessages.length !== messages.length + toolMessages.length) {
            this.logger.error('Current messages length does not match expected length', undefined, {
                current_length: currentMessages.length,
                expected_length: messages.length + toolMessages.length,
                original_length: messages.length,
                tool_messages_length: toolMessages.length,
            });
            return { messages: currentMessages };
        }
        return { messages: toolMessages };
    }
    async createAgentGraph(llm, tools, systemPrompt) {
        // Define the agent node
        const agentNode = async ({ messages }) => {
            // systemPrompt is already enhanced by beforeAction() if expectedOutputSchema was provided
            const systemMessage = new HumanMessage(systemPrompt);
            const allMessages = [systemMessage, ...messages];
            // Helper function for exponential backoff with jitter
            const exponentialBackoff = (attemptNumber) => {
                // Base delay: 1 second, exponentially increases (1s, 2s, 4s, 8s, ...)
                const baseDelay = AI_AGENT.RETRY_BASE_DELAY_MS;
                const maxDelay = AI_AGENT.RETRY_MAX_DELAY_MS;
                const delay = Math.min(baseDelay * Math.pow(AI_AGENT.RETRY_BACKOFF_MULTIPLIER, attemptNumber - 1), maxDelay);
                // Add jitter (random ±25% variation) to prevent thundering herd
                const jitter = delay * AI_AGENT.RETRY_JITTER_FACTOR * (Math.random() - 0.5);
                const finalDelay = delay + jitter;
                return new Promise((resolve) => setTimeout(resolve, finalDelay));
            };
            // Shared onFailedAttempt callback to avoid duplication
            const onFailedAttempt = async (error) => {
                const attemptNumber = error.attemptNumber;
                const retriesLeft = error.retriesLeft;
                // Check if this is a candidateContent error
                const errorMessage = error.message || String(error);
                if (errorMessage.includes('candidateContent') ||
                    errorMessage.includes('parts.reduce') ||
                    errorMessage.includes('undefined is not an object')) {
                    this.context?.logger?.error(`[AIAgent] Gemini candidateContent error detected (attempt ${attemptNumber}). This indicates blocked/empty content from Gemini API.`);
                }
                this.context?.logger?.warn(`[AIAgent] LLM call failed (attempt ${attemptNumber}/${this.params.model.maxRetries}). Retries left: ${retriesLeft}. Error: ${error.message}`);
                // Optionally emit streaming event for retry
                if (this.streamingCallback) {
                    await this.streamingCallback({
                        type: 'error',
                        data: {
                            error: `Retry attempt ${attemptNumber}/${this.params.model.maxRetries}: ${error.message}`,
                            recoverable: retriesLeft > 0,
                        },
                    });
                }
                // Wait with exponential backoff before retrying
                if (retriesLeft > 0) {
                    await exponentialBackoff(attemptNumber);
                }
            };
            // If we have tools, bind them to the LLM, then add retry logic
            // IMPORTANT: Must bind tools FIRST, then add retry - not the other way around
            const modelWithTools = tools.length > 0
                ? llm.bindTools(tools).withRetry({
                    stopAfterAttempt: this.params.model.maxRetries,
                    onFailedAttempt,
                })
                : llm.withRetry({
                    stopAfterAttempt: this.params.model.maxRetries,
                    onFailedAttempt,
                });
            try {
                // Use streaming if streamingCallback is provided
                if (this.streamingCallback) {
                    const messageId = `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
                    // Use invoke with callbacks for streaming
                    const response = await modelWithTools.invoke(allMessages, {
                        callbacks: [
                            {
                                handleLLMStart: async () => {
                                    await this.streamingCallback?.({
                                        type: 'llm_start',
                                        data: {
                                            model: this.params.model.model,
                                            temperature: this.params.model.temperature,
                                        },
                                    });
                                },
                                handleLLMEnd: async (output) => {
                                    // Extract thinking tokens from different model providers
                                    const thinking = extractAndStreamThinkingTokens(output);
                                    if (thinking) {
                                        await this.streamingCallback?.({
                                            type: 'think',
                                            data: {
                                                content: thinking,
                                                messageId,
                                            },
                                        });
                                    }
                                    const content = formatFinalResponse(generationsToMessageContent(output.generations.flat()), this.params.model.model).response;
                                    await this.streamingCallback?.({
                                        type: 'llm_complete',
                                        data: {
                                            messageId,
                                            content: content,
                                            totalTokens: output.llmOutput?.usage_metadata?.total_tokens,
                                        },
                                    });
                                },
                            },
                        ],
                    });
                    return { messages: [response] };
                }
                else {
                    // Non-streaming fallback
                    const response = await modelWithTools.invoke(allMessages);
                    return { messages: [response] };
                }
            }
            catch (error) {
                // Catch candidateContent errors that slip through SafeGeminiChat
                const errorMessage = error instanceof Error ? error.message : String(error);
                if (errorMessage.includes('candidateContent') ||
                    errorMessage.includes('parts.reduce') ||
                    errorMessage.includes('undefined is not an object')) {
                    this.logger.error('Gemini candidateContent error caught in agentNode', undefined, {
                        error_message: errorMessage,
                    });
                    // Return error as AIMessage instead of crashing
                    return {
                        messages: [
                            new AIMessage({
                                content: `[Gemini Error] Unable to generate response due to content filtering. Error: ${errorMessage}`,
                                additional_kwargs: {
                                    finishReason: 'ERROR',
                                    error: errorMessage,
                                },
                            }),
                        ],
                    };
                }
                // Rethrow other errors
                throw error;
            }
        };
        // Node that runs after agent to check afterLLMCall hook before routing
        const afterLLMCheckNode = async ({ messages, }) => {
            // Reset the flag at the start
            this.shouldContinueToAgent = false;
            // Get the last AI message
            const lastMessage = messages[messages.length - 1];
            const hasToolCalls = !!(lastMessage.tool_calls && lastMessage.tool_calls.length > 0);
            // Only call hook if we're about to end (no tool calls) and hook is provided
            if (!hasToolCalls && this.afterLLMCallHook) {
                this.logger.debug('No tool calls detected, calling afterLLMCall hook');
                const hookResult = await this.afterLLMCallHook({
                    messages,
                    lastAIMessage: lastMessage,
                    hasToolCalls,
                });
                // If hook wants to continue to agent, set flag and return modified messages
                if (hookResult.continueToAgent) {
                    this.logger.debug('AfterLLMCall hook requested retry to agent');
                    this.shouldContinueToAgent = true;
                    // Return the modified messages from the hook
                    // We need to return only the new messages to append
                    const newMessages = hookResult.messages.slice(messages.length);
                    return { messages: newMessages };
                }
            }
            // No modifications needed
            return { messages: [] };
        };
        // Define conditional edge function after LLM check
        const shouldContinueAfterLLMCheck = ({ messages, }) => {
            // First check if afterLLMCall hook requested continuing to agent
            if (this.shouldContinueToAgent) {
                return 'agent';
            }
            // Find the last AI message (could be followed by human messages from hook)
            const aiMessages = [];
            for (const msg of messages) {
                if (isAIMessage(msg)) {
                    aiMessages.push(msg);
                }
                else if ('tool_calls' in msg &&
                    msg.constructor?.name === 'AIMessageChunk') {
                    aiMessages.push(msg);
                }
            }
            const lastAIMessage = aiMessages[aiMessages.length - 1];
            // Check if the last AI message has tool calls
            if (lastAIMessage?.tool_calls && lastAIMessage.tool_calls.length > 0) {
                return 'tools';
            }
            return '__end__';
        };
        // Define conditional edge after tools to check if we should stop
        const shouldContinueAfterTools = () => {
            // Check if the afterToolCall hook requested stopping
            if (this.shouldStopAfterTools) {
                return '__end__';
            }
            // Otherwise continue back to agent
            return 'agent';
        };
        // Build the graph
        const graph = new StateGraph(MessagesAnnotation).addNode('agent', agentNode);
        if (tools.length > 0) {
            // Use custom tool node with hooks support
            const toolNode = async (state) => {
                return await this.executeToolsWithHooks(state, tools);
            };
            graph
                .addNode('tools', toolNode)
                .addNode('afterLLMCheck', afterLLMCheckNode)
                .addEdge('__start__', 'agent')
                .addEdge('agent', 'afterLLMCheck')
                .addConditionalEdges('afterLLMCheck', shouldContinueAfterLLMCheck)
                .addConditionalEdges('tools', shouldContinueAfterTools);
        }
        else {
            // Even without tools, add the afterLLMCheck node for hook support
            graph
                .addNode('afterLLMCheck', afterLLMCheckNode)
                .addEdge('__start__', 'agent')
                .addEdge('agent', 'afterLLMCheck')
                .addConditionalEdges('afterLLMCheck', shouldContinueAfterLLMCheck);
        }
        return graph.compile();
    }
    async executeAgent(graph, message, images, maxIterations, modelConfig, conversationHistory) {
        const jsonMode = modelConfig.jsonMode;
        const toolCalls = [];
        let iterations = 0;
        this.logger.info('Starting agent execution', {
            message_preview: message.substring(0, AI_AGENT.MESSAGE_PREVIEW_LENGTH) + '...',
            max_iterations: maxIterations,
            model: modelConfig.model,
        });
        try {
            this.logger.debug('Invoking agent graph');
            // Build messages array starting with conversation history (for KV cache optimization)
            const initialMessages = [];
            // Convert conversation history to LangChain messages if provided
            // This enables KV cache optimization by keeping previous turns as separate messages
            if (conversationHistory && conversationHistory.length > 0) {
                for (const historyMsg of conversationHistory) {
                    switch (historyMsg.role) {
                        case 'user':
                            initialMessages.push(new HumanMessage(historyMsg.content));
                            break;
                        case 'assistant':
                            initialMessages.push(new AIMessage(historyMsg.content));
                            break;
                        case 'tool':
                            // Tool messages require a tool_call_id
                            if (historyMsg.toolCallId) {
                                initialMessages.push(new ToolMessage({
                                    content: historyMsg.content,
                                    tool_call_id: historyMsg.toolCallId,
                                    name: historyMsg.name,
                                }));
                            }
                            break;
                    }
                }
            }
            // Create the current human message with text and optional images
            let humanMessage;
            if (images && images.length > 0) {
                this.logger.debug('Creating multimodal message', {
                    image_count: images.length,
                });
                // Create multimodal content array
                const content = [{ type: 'text', text: message }];
                // Add images to content
                for (const image of images) {
                    let imageUrl;
                    if (image.type === 'base64') {
                        // Base64 encoded image
                        imageUrl = `data:${image.mimeType};base64,${image.data}`;
                    }
                    else {
                        // URL image - fetch and convert to base64 for Google Gemini compatibility
                        try {
                            this.logger.debug('Fetching image from URL', { url: image.url });
                            // SECURITY FIX: Add timeout to prevent hanging on malicious URLs
                            const controller = new AbortController();
                            const timeoutId = setTimeout(() => controller.abort(), AI_AGENT.IMAGE_FETCH_TIMEOUT_MS);
                            const response = await fetch(image.url, {
                                signal: controller.signal,
                                // SECURITY: Don't follow redirects to prevent SSRF via redirect chains
                                redirect: 'manual',
                            });
                            clearTimeout(timeoutId);
                            if (!response.ok) {
                                throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
                            }
                            // SECURITY: Validate content type to prevent XSS
                            const contentType = response.headers.get('content-type') || 'image/png';
                            if (!contentType.startsWith('image/')) {
                                throw new Error(`Invalid content type: ${contentType}. Only image types are allowed.`);
                            }
                            // SECURITY: Limit file size to prevent DoS
                            const contentLength = response.headers.get('content-length');
                            if (contentLength && parseInt(contentLength) > AI_AGENT.IMAGE_MAX_SIZE_BYTES) {
                                throw new Error(`Image too large: ${contentLength} bytes. Maximum size is ${AI_AGENT.IMAGE_MAX_SIZE_BYTES / 1024 / 1024}MB.`);
                            }
                            const arrayBuffer = await response.arrayBuffer();
                            // Double-check size after download
                            if (arrayBuffer.byteLength > AI_AGENT.IMAGE_MAX_SIZE_BYTES) {
                                throw new Error(`Downloaded image too large: ${arrayBuffer.byteLength} bytes. Maximum size is ${AI_AGENT.IMAGE_MAX_SIZE_BYTES / 1024 / 1024}MB.`);
                            }
                            const base64Data = Buffer.from(arrayBuffer).toString('base64');
                            imageUrl = `data:${contentType};base64,${base64Data}`;
                            this.logger.debug('Successfully converted URL image to base64');
                        }
                        catch (error) {
                            this.logger.error('Error fetching image from URL', error, { url: image.url });
                            throw new Error(`Failed to load image from URL ${image.url}: ${error instanceof Error ? error.message : 'Unknown error'}`);
                        }
                    }
                    content.push({
                        type: 'image_url',
                        image_url: { url: imageUrl },
                    });
                    // Add image description if provided
                    if (image.description) {
                        content.push({
                            type: 'text',
                            text: `Image description: ${image.description}`,
                        });
                    }
                }
                humanMessage = new HumanMessage({ content });
            }
            else {
                // Text-only message
                humanMessage = new HumanMessage(message);
            }
            // Add the current message to the conversation
            initialMessages.push(humanMessage);
            const result = await graph.invoke({ messages: initialMessages }, { recursionLimit: maxIterations });
            this.logger.info('Graph execution completed', {
                total_messages: result.messages.length,
            });
            iterations = result.messages.length;
            // Extract tool calls from messages and track individual LLM calls
            // Store tool calls temporarily to match with their responses
            const toolCallMap = new Map();
            for (let i = 0; i < result.messages.length; i++) {
                const msg = result.messages[i];
                if (msg instanceof AIMessage ||
                    (msg instanceof AIMessageChunk && msg.tool_calls)) {
                    const typedToolCalls = msg.tool_calls;
                    // Log and track tool calls
                    for (const toolCall of typedToolCalls || []) {
                        toolCallMap.set(toolCall.id, {
                            name: toolCall.name,
                            args: toolCall.args,
                        });
                        this.logger.debug('Tool call initiated', {
                            tool_name: toolCall.name,
                            args: toolCall.args,
                        });
                    }
                }
                else if (msg instanceof ToolMessage) {
                    // Match tool response to its call
                    const toolCall = toolCallMap.get(msg.tool_call_id);
                    if (toolCall) {
                        // Parse content if it's a JSON string
                        let output = msg.content;
                        if (typeof output === 'string') {
                            try {
                                output = JSON.parse(output);
                            }
                            catch {
                                // Keep as string if not valid JSON
                            }
                        }
                        this.logger.debug('Tool output received', {
                            output_preview: typeof output === 'string'
                                ? output.substring(0, AI_AGENT.MESSAGE_PREVIEW_LENGTH) + '...'
                                : JSON.stringify(output).substring(0, AI_AGENT.MESSAGE_PREVIEW_LENGTH) + '...',
                        });
                        toolCalls.push({
                            tool: toolCall.name,
                            input: toolCall.args,
                            output,
                        });
                    }
                }
            }
            // Get the final AI message response
            this.logger.debug('Filtering AI messages');
            const aiMessages = result.messages.filter((msg) => isAIMessage(msg) || isAIMessageChunk(msg));
            this.logger.debug('AI messages found', {
                count: aiMessages.length,
            });
            const finalMessage = aiMessages[aiMessages.length - 1];
            if (finalMessage?.additional_kwargs?.finishReason === 'SAFETY_BLOCKED') {
                throw new Error(`[Gemini Error] Unable to generate a response. Please increase maxTokens in model configuration or try again with a different model.`);
            }
            // Check for MAX_TOKENS finish reason
            if (finalMessage?.additional_kwargs?.finishReason === 'MAX_TOKENS') {
                throw new Error('Response was truncated due to max tokens limit. Please increase maxTokens in model configuration.');
            }
            // Track token usage from ALL AI messages (not just the final one)
            // This is critical for multi-iteration workflows where the agent calls tools multiple times
            let totalInputTokens = 0;
            let totalOutputTokens = 0;
            let totalTokensSum = 0;
            for (const msg of result.messages) {
                if (msg instanceof AIMessage ||
                    (msg instanceof AIMessageChunk && msg.usage_metadata)) {
                    totalInputTokens +=
                        msg.usage_metadata?.input_tokens ||
                            0;
                    totalOutputTokens +=
                        msg.usage_metadata?.output_tokens ||
                            0;
                    totalTokensSum +=
                        msg.usage_metadata?.total_tokens ||
                            0;
                }
            }
            if (totalTokensSum > 0 && this.context && this.context.logger) {
                this.context.logger.logTokenUsage({
                    usage: totalInputTokens,
                    service: this.getCredentialTypeForModel(modelConfig.model),
                    unit: 'input_tokens',
                    subService: modelConfig.model,
                }, `LLM completion: ${totalInputTokens} input`, {
                    bubbleName: 'ai-agent',
                    variableId: this.context?.variableId,
                    operationType: 'bubble_execution',
                });
                this.context.logger.logTokenUsage({
                    usage: totalOutputTokens,
                    service: this.getCredentialTypeForModel(modelConfig.model),
                    unit: 'output_tokens',
                    subService: modelConfig.model,
                }, `LLM completion: ${totalOutputTokens} output`, {
                    bubbleName: 'ai-agent',
                    variableId: this.context?.variableId,
                    operationType: 'bubble_execution',
                });
            }
            const response = finalMessage?.content || '';
            // Use shared formatting method
            const formattedResult = formatFinalResponse(response, modelConfig.model, jsonMode);
            // If there's an error from formatting (e.g., invalid JSON), return early
            if (formattedResult.error) {
                return {
                    response: formattedResult.response,
                    toolCalls: toolCalls.length > 0 ? toolCalls : [],
                    iterations,
                    error: formattedResult.error,
                    success: false,
                };
            }
            const finalResponse = formattedResult.response;
            this.logger.info('Agent execution completed', {
                response_length: typeof finalResponse === 'string'
                    ? finalResponse.length
                    : JSON.stringify(finalResponse).length,
                tool_calls_count: toolCalls.length,
                iterations,
            });
            return {
                response: typeof finalResponse === 'string'
                    ? finalResponse
                    : JSON.stringify(finalResponse),
                toolCalls: toolCalls.length > 0 ? toolCalls : [],
                iterations,
                error: '',
                success: true,
            };
        }
        catch (error) {
            this.logger.warn('Execution error (continuing)', {
                error,
                tool_calls_before_error: toolCalls.length,
                iterations_before_error: iterations,
            });
            // Model fallback logic - only retry if this config has a backup model
            if (modelConfig.backupModel) {
                this.logger.info('Retrying with backup model', {
                    backup_model: modelConfig.backupModel.model,
                    primary_model: modelConfig.model,
                });
                this.context?.logger?.warn(`Primary model ${modelConfig.model} failed: ${error instanceof Error ? error.message : 'Unknown error'}. Retrying with backup model... ${modelConfig.backupModel.model}`);
                this.streamingCallback?.({
                    type: 'error',
                    data: {
                        error: `Primary model ${modelConfig.model} failed: ${error instanceof Error ? error.message : 'Unknown error'}. Retrying with backup model... ${modelConfig.backupModel.model}`,
                        recoverable: true,
                    },
                });
                const backupModelConfig = this.buildModelConfig(modelConfig, modelConfig.backupModel);
                const backupResult = await this.executeWithModel(backupModelConfig);
                return backupResult;
            }
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            // Return partial results to allow execution to continue
            // Include any tool calls that were completed before the error
            return {
                response: `Execution error: ${errorMessage}`,
                success: false, // Still false but don't completely halt execution
                iterations,
                toolCalls: toolCalls.length > 0 ? toolCalls : [], // Preserve completed tool calls
                error: errorMessage,
            };
        }
    }
}
//# sourceMappingURL=ai-agent.js.map