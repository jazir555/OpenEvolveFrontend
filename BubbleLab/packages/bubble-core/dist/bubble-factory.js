import { BUBBLE_CREDENTIAL_OPTIONS, } from '@bubblelab/shared-schemas';
import { WebCrawlTool } from './bubbles/tool-bubble/web-crawl-tool.js';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildClassNameLookup as buildLookupForSource, parseBubbleInstancesFromSource, } from './utils/source-bubble-parser.js';
export class BubbleFactory {
    registry = new Map();
    static dependenciesPopulated = false;
    static detailedDepsCache = new Map();
    // Stores detailed dependencies inferred from source for each registered bubble
    detailedDeps = new Map();
    constructor(autoRegisterDefaults = false) {
        if (autoRegisterDefaults) {
            this.registerDefaults();
        }
        // Seed instance detailed deps from global cache if available
        if (BubbleFactory.detailedDepsCache.size > 0) {
            for (const [name, deps] of BubbleFactory.detailedDepsCache) {
                this.detailedDeps.set(name, deps);
            }
        }
    }
    /**
     * Register a bubble class with the factory
     */
    register(name, bubbleClass) {
        if (this.registry.has(name)) {
            // Silently skip if already registered - makes it idempotent
            return;
        }
        this.registry.set(name, bubbleClass);
    }
    /**
     * Get a bubble class from the registry
     */
    get(name) {
        return this.registry.get(name);
    }
    /**
     * Create a bubble instance
     */
    createBubble(name, params, context) {
        const BubbleClass = this.registry.get(name);
        if (!BubbleClass) {
            throw new Error(`Bubble '${name}' not found in factory registry`);
        }
        // Always pass params, even if undefined
        return new BubbleClass(params, context);
    }
    getDetailedDependencies(name) {
        return this.detailedDeps.get(name) || [];
    }
    /**
     * List all registered bubble names
     */
    list() {
        return Array.from(this.registry.keys());
    }
    // Return a list of bubbles to be used in the BubbleFlow code generator
    listBubblesForCodeGenerator() {
        return [
            'postgresql',
            'ai-agent',
            'slack',
            'telegram',
            'resend',
            'google-drive',
            'gmail',
            'google-sheets',
            'google-calendar',
            'pdf-form-operations',
            'slack-formatter-agent',
            'research-agent-tool',
            'web-crawl-tool',
            'web-scrape-tool',
            'web-search-tool',
            'reddit-scrape-tool',
            'apify',
            'instagram-tool',
            'linkedin-tool',
            'tiktok-tool',
            'twitter-tool',
            'google-maps-tool',
            'youtube-tool',
            'github',
            'eleven-labs',
            'followupboss',
            'agi-inc',
            'airtable',
            'notion',
            'firecrawl',
            'insforge-db',
        ];
    }
    async registerDefaults() {
        // Import and register all default bubbles
        // This will be implemented in a separate file to avoid circular deps
        // Register all default bubbles
        const { HelloWorldBubble } = await import('./bubbles/service-bubble/hello-world.js');
        const { AIAgentBubble } = await import('./bubbles/service-bubble/ai-agent.js');
        const { PostgreSQLBubble } = await import('./bubbles/service-bubble/postgresql.js');
        const { SlackBubble } = await import('./bubbles/service-bubble/slack.js');
        const { TelegramBubble } = await import('./bubbles/service-bubble/telegram.js');
        const { ResendBubble } = await import('./bubbles/service-bubble/resend.js');
        const { HttpBubble } = await import('./bubbles/service-bubble/http.js');
        const { StorageBubble } = await import('./bubbles/service-bubble/storage.js');
        const { GoogleDriveBubble } = await import('./bubbles/service-bubble/google-drive.js');
        const { GmailBubble } = await import('./bubbles/service-bubble/gmail.js');
        const { GoogleSheetsBubble } = await import('./bubbles/service-bubble/google-sheets');
        const { GoogleCalendarBubble } = await import('./bubbles/service-bubble/google-calendar.js');
        const { ApifyBubble } = await import('./bubbles/service-bubble/apify');
        const { GithubBubble } = await import('./bubbles/service-bubble/github.js');
        const { FollowUpBossBubble } = await import('./bubbles/service-bubble/followupboss.js');
        const { NotionBubble } = await import('./bubbles/service-bubble/notion/notion.js');
        const { DatabaseAnalyzerWorkflowBubble } = await import('./bubbles/workflow-bubble/database-analyzer.workflow.js');
        const { SlackNotifierWorkflowBubble } = await import('./bubbles/workflow-bubble/slack-notifier.workflow.js');
        const { SlackDataAssistantWorkflow } = await import('./bubbles/workflow-bubble/slack-data-assistant.workflow.js');
        const { ListBubblesTool } = await import('./bubbles/tool-bubble/list-bubbles-tool.js');
        const { GetBubbleDetailsTool } = await import('./bubbles/tool-bubble/get-bubble-details-tool.js');
        const { SQLQueryTool } = await import('./bubbles/tool-bubble/sql-query-tool.js');
        const { ChartJSTool } = await import('./bubbles/tool-bubble/chart-js-tool.js');
        const { BubbleFlowValidationTool } = await import('./bubbles/tool-bubble/bubbleflow-validation-tool.js');
        const { EditBubbleFlowTool } = await import('./bubbles/tool-bubble/code-edit-tool.js');
        const { WebSearchTool } = await import('./bubbles/tool-bubble/web-search-tool.js');
        const { WebScrapeTool } = await import('./bubbles/tool-bubble/web-scrape-tool.js');
        const { WebExtractTool } = await import('./bubbles/tool-bubble/web-extract-tool.js');
        const { ResearchAgentTool } = await import('./bubbles/tool-bubble/research-agent-tool.js');
        const { RedditScrapeTool } = await import('./bubbles/tool-bubble/reddit-scrape-tool.js');
        const { InstagramTool } = await import('./bubbles/tool-bubble/instagram-tool.js');
        const { LinkedInTool } = await import('./bubbles/tool-bubble/linkedin-tool.js');
        const { YouTubeTool } = await import('./bubbles/tool-bubble/youtube-tool.js');
        const { TikTokTool } = await import('./bubbles/tool-bubble/tiktok-tool.js');
        const { TwitterTool } = await import('./bubbles/tool-bubble/twitter-tool.js');
        const { GoogleMapsTool } = await import('./bubbles/tool-bubble/google-maps-tool.js');
        const { SlackFormatterAgentBubble } = await import('./bubbles/workflow-bubble/slack-formatter-agent.js');
        const { PDFFormOperationsWorkflow } = await import('./bubbles/workflow-bubble/pdf-form-operations.workflow.js');
        const { PDFOcrWorkflow } = await import('./bubbles/workflow-bubble/pdf-ocr.workflow.js');
        const { GenerateDocumentWorkflow } = await import('./bubbles/workflow-bubble/generate-document.workflow.js');
        const { ParseDocumentWorkflow } = await import('./bubbles/workflow-bubble/parse-document.workflow.js');
        const { ElevenLabsBubble } = await import('./bubbles/service-bubble/eleven-labs.js');
        const { AGIIncBubble } = await import('./bubbles/service-bubble/agi-inc.js');
        const { AirtableBubble } = await import('./bubbles/service-bubble/airtable.js');
        const { FirecrawlBubble } = await import('./bubbles/service-bubble/firecrawl.js');
        const { InsForgeDbBubble } = await import('./bubbles/service-bubble/insforge-db.js');
        // Create the default factory instance
        this.register('hello-world', HelloWorldBubble);
        this.register('ai-agent', AIAgentBubble);
        this.register('postgresql', PostgreSQLBubble);
        this.register('slack', SlackBubble);
        this.register('telegram', TelegramBubble);
        this.register('resend', ResendBubble);
        this.register('http', HttpBubble);
        this.register('storage', StorageBubble);
        this.register('google-drive', GoogleDriveBubble);
        this.register('gmail', GmailBubble);
        this.register('google-sheets', GoogleSheetsBubble);
        this.register('google-calendar', GoogleCalendarBubble);
        this.register('apify', ApifyBubble);
        this.register('github', GithubBubble);
        this.register('followupboss', FollowUpBossBubble);
        this.register('notion', NotionBubble);
        this.register('database-analyzer', DatabaseAnalyzerWorkflowBubble);
        this.register('slack-notifier', SlackNotifierWorkflowBubble);
        this.register('slack-data-assistant', SlackDataAssistantWorkflow);
        this.register('slack-formatter-agent', SlackFormatterAgentBubble);
        this.register('pdf-form-operations', PDFFormOperationsWorkflow);
        this.register('pdf-ocr-workflow', PDFOcrWorkflow);
        this.register('generate-document-workflow', GenerateDocumentWorkflow);
        this.register('parse-document-workflow', ParseDocumentWorkflow);
        this.register('get-bubble-details-tool', GetBubbleDetailsTool);
        this.register('list-bubbles-tool', ListBubblesTool);
        this.register('sql-query-tool', SQLQueryTool);
        this.register('chart-js-tool', ChartJSTool);
        this.register('bubbleflow-validation-tool', BubbleFlowValidationTool);
        this.register('code-edit-tool', EditBubbleFlowTool);
        this.register('web-search-tool', WebSearchTool);
        this.register('web-scrape-tool', WebScrapeTool);
        this.register('web-extract-tool', WebExtractTool);
        this.register('research-agent-tool', ResearchAgentTool);
        this.register('reddit-scrape-tool', RedditScrapeTool);
        this.register('instagram-tool', InstagramTool);
        this.register('linkedin-tool', LinkedInTool);
        this.register('tiktok-tool', TikTokTool);
        this.register('twitter-tool', TwitterTool);
        this.register('google-maps-tool', GoogleMapsTool);
        this.register('youtube-tool', YouTubeTool);
        this.register('web-crawl-tool', WebCrawlTool);
        this.register('eleven-labs', ElevenLabsBubble);
        this.register('agi-inc', AGIIncBubble);
        this.register('airtable', AirtableBubble);
        this.register('firecrawl', FirecrawlBubble);
        this.register('insforge-db', InsForgeDbBubble);
        // After all default bubbles are registered, auto-populate bubbleDependencies
        if (!BubbleFactory.dependenciesPopulated) {
            console.log('Populating bubble dependencies from source....');
            await this.populateBubbleDependenciesFromSource();
            BubbleFactory.dependenciesPopulated = true;
            // Cache detailed dependencies globally for seeding future instances
            BubbleFactory.detailedDepsCache = new Map(this.detailedDeps);
        }
        else {
            // Seed this instance from the global cache if available
            if (BubbleFactory.detailedDepsCache.size > 0) {
                for (const [name, deps] of BubbleFactory.detailedDepsCache) {
                    this.detailedDeps.set(name, deps);
                }
            }
        }
    }
    /**
     * Get all registered bubble classes
     */
    getAll() {
        return Array.from(this.registry.values());
    }
    /**
     * Get metadata for a bubble without instantiating it
     */
    getMetadata(name) {
        const BubbleClass = this.get(name);
        if (!BubbleClass)
            return undefined;
        // Type guard to check if schema is a ZodObject
        const schemaParams = BubbleClass.schema &&
            typeof BubbleClass.schema === 'object' &&
            'shape' in BubbleClass.schema
            ? BubbleClass.schema.shape
            : undefined;
        return {
            bubbleDependenciesDetailed: this.detailedDeps.get(BubbleClass.bubbleName),
            name: BubbleClass.bubbleName,
            shortDescription: BubbleClass.shortDescription,
            longDescription: BubbleClass.longDescription,
            alias: BubbleClass.alias,
            credentialOptions: BubbleClass.credentialOptions,
            bubbleDependencies: BubbleClass.bubbleDependencies,
            // Provide richer dependency details (ai-agent may include tools)
            schema: BubbleClass.schema,
            resultSchema: BubbleClass.resultSchema,
            type: BubbleClass.type,
            params: schemaParams,
        };
    }
    /**
     * Get all bubble metadata
     */
    getAllMetadata() {
        return this.list()
            .map((name) => this.getMetadata(name))
            .filter(Boolean);
    }
    /**
     * Scan bubble source modules to infer direct dependencies between bubbles by
     * inspecting ES module import statements, then attach the resulting
     * `bubbleDependencies` array onto the corresponding registered classes.
     *
     * Notes:
     * - Works in both dev (src) and build (dist) because it resolves paths
     *   relative to this module at runtime.
     * - Only imports under ./bubbles/** that themselves define a bubble class are
     *   considered dependencies; all other imports are ignored.
     */
    async populateBubbleDependenciesFromSource() {
        try {
            const currentFilePath = fileURLToPath(import.meta.url);
            const baseDir = path.dirname(currentFilePath);
            const bubblesDir = path.resolve(baseDir, './bubbles');
            console.log('Bubbles directory:', bubblesDir);
            // Gather all .js and .ts files under bubbles/**
            const bubbleFiles = await this.listModuleFilesRecursively(bubblesDir);
            // Build lookup once for all files
            const lookup = buildLookupForSource(this.registry);
            for (const filePath of bubbleFiles) {
                const content = await fs.readFile(filePath, 'utf-8');
                const ownerBubbleNames = this.extractBubbleNamesFromContent(content);
                if (ownerBubbleNames.length === 0) {
                    continue;
                }
                // Parse instances used within this file
                let instancesByDep = new Map();
                try {
                    instancesByDep = parseBubbleInstancesFromSource(content, lookup, {
                        debug: false,
                        filePath,
                    });
                }
                catch {
                    // ignore parser failures for this file
                }
                // Collect ai-agent tools from instances directly (AST-derived)
                const aiAgentInst = instancesByDep.get('ai-agent');
                const aiTools = Array.from(new Set((aiAgentInst || [])
                    .flatMap((i) => i.tools || [])
                    .filter((t) => typeof t === 'string')));
                for (const owner of ownerBubbleNames) {
                    const detailed = [];
                    for (const [depName, instList] of instancesByDep.entries()) {
                        if (depName === owner)
                            continue;
                        const spec = {
                            name: depName,
                            instances: instList.map((i) => ({
                                variableName: i.variableName,
                                isAnonymous: i.isAnonymous,
                                startLine: i.startLine,
                                endLine: i.endLine,
                            })),
                        };
                        if (depName === 'ai-agent' && aiTools.length > 0) {
                            spec.tools = aiTools;
                        }
                        detailed.push(spec);
                    }
                    // Persist results for this owner bubble
                    this.detailedDeps.set(owner, detailed);
                    // Maintain classic flat dependency list on the class
                    const klass = this.get(owner);
                    if (klass) {
                        try {
                            klass.bubbleDependencies = detailed.map((d) => d.name);
                        }
                        catch {
                            try {
                                Object.defineProperty(klass, 'bubbleDependencies', {
                                    value: detailed.map((d) => d.name),
                                    configurable: true,
                                });
                            }
                            catch {
                                // ignore
                            }
                        }
                    }
                }
            }
        }
        catch {
            // Silently ignore issues in dependency scanning to avoid blocking runtime
        }
    }
    async listModuleFilesRecursively(dir) {
        const out = [];
        const entries = await fs.readdir(dir, { withFileTypes: true });
        for (const entry of entries) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                const nested = await this.listModuleFilesRecursively(full);
                out.push(...nested);
            }
            else if (entry.isFile() &&
                (full.endsWith('.ts') || full.endsWith('.js')) &&
                !full.endsWith('.test.ts') &&
                !full.endsWith('.d.ts')) {
                out.push(full);
            }
        }
        return out;
    }
    extractBubbleNamesFromContent(content) {
        const names = [];
        // Look for static bubbleName definitions in the class body
        const nameRegex = /static\s+(?:readonly\s+)?bubbleName\s*(?::[^=]+)?=\s*['"]([^'"\n]+)['"]/g;
        let match;
        while ((match = nameRegex.exec(content)) !== null) {
            names.push(match[1]);
        }
        return names;
    }
    /**
     * Get credential to bubble name mapping from registered bubbles
     * Provides type-safe mapping based on actual registered bubbles
     */
    getCredentialToBubbleMapping() {
        const mapping = {};
        for (const [bubbleName, credentialOptions] of Object.entries(BUBBLE_CREDENTIAL_OPTIONS)) {
            // Get the bubble class to check its type
            const BubbleClass = this.get(bubbleName);
            // Only include service bubbles for credential validation
            if (BubbleClass && BubbleClass.type === 'service') {
                for (const credentialType of credentialOptions) {
                    // Only map if we haven't seen this credential type before
                    // This gives priority to the first service bubble for each credential
                    if (!mapping[credentialType]) {
                        mapping[credentialType] = bubbleName;
                    }
                }
            }
        }
        return mapping;
    }
    /**
     * Get bubble name for a specific credential type
     */
    getBubbleNameForCredential(credentialType) {
        const mapping = this.getCredentialToBubbleMapping();
        return mapping[credentialType];
    }
    /**
     * Check if a credential type is supported by any registered bubble
     */
    isCredentialSupported(credentialType) {
        return this.getBubbleNameForCredential(credentialType) !== undefined;
    }
    /**
     * Generate comprehensive BubbleFlow boilerplate template with all imports
     * This template includes ALL available bubble classes and types
     * Perfect for AI code generation and testing
     */
    generateBubbleFlowBoilerplate(options) {
        const className = options?.className || 'GeneratedFlow';
        return `
import {z} from 'zod';
import {
  // Base classes
  BubbleFlow,

  // Service Bubbles (Connects to external services)
  HelloWorldBubble, // bubble name: 'hello-world'
  AIAgentBubble, // bubble name: 'ai-agent'
  PostgreSQLBubble, // bubble name: 'postgresql'
  SlackBubble, // bubble name: 'slack'
  ResendBubble, // bubble name: 'resend'
  GoogleDriveBubble, // bubble name: 'google-drive'
  GoogleSheetsBubble, // bubble name: 'google-sheets'
  GoogleCalendarBubble, // bubble name: 'google-calendar'
  GmailBubble, // bubble name: 'gmail'
  SlackFormatterAgentBubble, // bubble name: 'slack-formatter-agent'
  HttpBubble, // bubble name: 'http'
  StorageBubble, // bubble name: 'storage'
  ApifyBubble, // bubble name: 'apify'
  ElevenLabsBubble, // bubble name: 'eleven-labs'
  FollowUpBossBubble, // bubble name: 'followupboss'

  // Tool Bubbles (Perform useful actions)
  ResearchAgentTool, // bubble name: 'research-agent-tool'
  RedditScrapeTool, // bubble name: 'reddit-scrape-tool'
  WebScrapeTool, // bubble name: 'web-scrape-tool'
  WebCrawlTool, // bubble name: 'web-crawl-tool'
  WebSearchTool, // bubble name: 'web-search-tool'
  InstagramTool, // bubble name: 'instagram-tool'
  LinkedInTool, // bubble name: 'linkedin-tool'
  TikTokTool, // bubble name: 'tiktok-tool'
  TwitterTool, // bubble name: 'twitter-tool'
  GoogleMapsTool, // bubble name: 'google-maps-tool'
  YouTubeTool, // bubble name: 'youtube-tool'

  // Event Types (How the workflow is triggered)
  type WebhookEvent,
  type CronEvent,
} from '@bubblelab/bubble-core';

export interface Output {
  // TODO: Add your output fields here
  message: string;
  processed: boolean;
}

// TRIGGER TYPE 1: Webhook HTTP Trigger
// Define your custom input interface for webhook triggers
export interface CustomWebhookPayload extends WebhookEvent {
  // TODO: Add your custom payload fields here
  input?: string;
}

export class ${className} extends BubbleFlow<'webhook/http'> {
  
  // Sanitizes and normalizes raw webhook input by trimming whitespace and converting to uppercase
  private transformData(input: string | undefined) {
    // Example: Transform or clean the input data
    if (!input || input.trim().length === 0) return null;
    return input.trim().toUpperCase();
  }

  // Sends cleaned input to AI agent for natural language processing and response generation
  // Condition: Only runs when transformedInput is not null and has more than 3 characters
  private async processWithAI(input: string) {
    const agent = new AIAgentBubble({
      model: { model: 'google/gemini-2.5-flash' },
      systemPrompt: 'You are a helpful assistant.',
      message: \`Process this input: \${input}\`
    });

    const result = await agent.action();

    if (!result.success) {
      throw new Error(\`AI Agent failed: \${result.error}\`);
    }

    return result.data.response;
  }

  // Constructs final output payload with either AI-generated response or default fallback message
  private formatOutput(response: string | null, wasProcessed: boolean) {
    return {
      message: response || 'No input provided',
      processed: wasProcessed,
    };
  }

  // Main workflow orchestration
  // - No Bubbles directly in handle()
  // - No try/catch in handle() (let errors bubble up)
  // - Only calls to private methods
  async handle(payload: CustomWebhookPayload): Promise<Output> {
    const transformedInput = this.transformData(payload.input);

    // Only process with AI if input meets minimum length requirement
    let aiResponse: string | null = null;
    if (transformedInput && transformedInput.length > 3) {
      aiResponse = await this.processWithAI(transformedInput);
    }

    return this.formatOutput(aiResponse, aiResponse !== null);
  }
}

// TRIGGER TYPE 2: Cron Schedule Trigger
// For cron-based scheduled workflows, or any workflow that can be benefited from being scheduled, extend BubbleFlow with 'schedule/cron'
// and define the cronSchedule property with a cron expression
// Time is in utc timezone, so if you want to schedule for a specific timezone, you need to convert the timezone to utc before writing the cron expression
/*
export interface CustomCronPayload extends CronEvent {
  // TODO: Add your custom payload fields here
}

export class ${className}Cron extends BubbleFlow<'schedule/cron'> {
  // Define cron schedule using standard 5-part cron format:
  // * * * * * = minute hour day-of-month month day-of-week
  // Examples:
  //   '0 0 * * *'     = Daily at midnight
  //   '0 9 * * 1-5'   = Every weekday at 9am
  //   '*/15 * * * *' = Every 15 minutes
  //   '0 0 1 * *'     = First day of every month at midnight
  readonly cronSchedule = '* 3 * * * *'; // Every 3 minutes

  // Performs scheduled database check or external API call to fetch latest data
  private async performScheduledTask() {
     // Example: Check a database or API
     return "Task completed";
  }

  async handle(payload: CustomCronPayload): Promise<Output> {
    const result = await this.performScheduledTask();

    return { message: result, processed: true };
  }
}
*/`;
    }
}
//# sourceMappingURL=bubble-factory.js.map