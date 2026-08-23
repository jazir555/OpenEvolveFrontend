import { z } from 'zod';
import type { IBubble, BubbleContext } from './types/bubble.js';
import {
  CredentialType,
  type BubbleName,
  type BubbleNodeType,
  BUBBLE_CREDENTIAL_OPTIONS,
  TRIGGER_EVENT_CONFIGS,
} from '@bubblelab/shared-schemas';
// Local type to describe detailed dependencies without cross-package type coupling
type BubbleDependencySpec = {
  name: BubbleName;
  tools?: BubbleName[];
  instances?: Array<{
    variableName: string;
    isAnonymous: boolean;
    startLine?: number;
    endLine?: number;
  }>;
};
import type { LangGraphTool } from './types/tool-bubble-class.js';
import { WebCrawlTool } from './bubbles/tool-bubble/web-crawl-tool.js';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildClassNameLookup as buildLookupForSource,
  parseBubbleInstancesFromSource,
} from './utils/source-bubble-parser.js';

// Type for concrete bubble class constructors with static metadata
export type BubbleClassWithMetadata<TResult extends object = object> = {
  new (
    params: unknown,
    context?: BubbleContext
  ): IBubble<
    {
      success: boolean;
      error: string;
    } & TResult
  >;
  readonly bubbleName: BubbleName;
  readonly schema:
    | z.ZodObject<z.ZodRawShape>
    | z.ZodDiscriminatedUnion<string, z.ZodObject<z.ZodRawShape>[]>;
  readonly resultSchema?:
    | z.ZodObject<z.ZodRawShape>
    | z.ZodDiscriminatedUnion<string, z.ZodObject<z.ZodRawShape>[]>;
  readonly shortDescription: string;
  readonly longDescription: string;
  readonly alias?: string;
  readonly type: BubbleNodeType;
  readonly credentialOptions?: CredentialType[];
  readonly bubbleDependencies?: BubbleName[];
  toolAgent?: (
    credentials: Partial<Record<CredentialType, string>>,
    config?: Record<string, unknown>,
    context?: BubbleContext
  ) => LangGraphTool;
};

export class BubbleFactory {
  private registry = new Map<BubbleName, BubbleClassWithMetadata<any>>();
  private static dependenciesPopulated = false;
  private static optionalImportFailures = new Set<string>();
  private static detailedDepsCache = new Map<
    BubbleName,
    BubbleDependencySpec[]
  >();
  // Stores detailed dependencies inferred from source for each registered bubble
  private detailedDeps = new Map<BubbleName, BubbleDependencySpec[]>();

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

  private static _instance: BubbleFactory | null = null;
  private static _ready: Promise<void> | null = null;

  /**
   * Returns a singleton BubbleFactory with all default bubbles registered.
   * Awaits registration so callers can immediately `createBubble(...).action()`.
   */
  static async getInstance(): Promise<BubbleFactory> {
    if (!BubbleFactory._instance) {
      BubbleFactory._instance = new BubbleFactory(false);
      BubbleFactory._ready = BubbleFactory._instance.registerDefaults();
    }
    await BubbleFactory._ready;
    return BubbleFactory._instance;
  }

  /**
   * Register a bubble class with the factory
   */
  register(
    name: BubbleName,
    bubbleClass?: BubbleClassWithMetadata<any> | null
  ): void {
    if (!bubbleClass) {
      return;
    }
    if (this.registry.has(name)) {
      // Silently skip if already registered - makes it idempotent
      return;
    }
    this.registry.set(name, bubbleClass);
  }

  private async safeImport<T extends Record<string, unknown>>(
    modulePath: string
  ): Promise<T | null> {
    try {
      return (await import(modulePath)) as T;
    } catch (error) {
      const message =
        typeof error === 'object' && error && 'message' in error
          ? String((error as { message?: unknown }).message)
          : String(error);
      if (!BubbleFactory.optionalImportFailures.has(modulePath)) {
        console.warn(
          `[BubbleFactory] Skipping optional bubble module '${modulePath}': ${message}`
        );
        BubbleFactory.optionalImportFailures.add(modulePath);
      }
      return null;
    }
  }

  /**
   * Get a bubble class from the registry
   */
  get(name: BubbleName): BubbleClassWithMetadata<any> | undefined {
    return this.registry.get(name as BubbleName);
  }

  /**
   * Create a bubble instance
   */
  createBubble<T extends IBubble = IBubble>(
    name: BubbleName,
    params?: unknown,
    context?: BubbleContext
  ): T {
    const BubbleClass = this.registry.get(name as BubbleName);
    if (!BubbleClass) {
      throw new Error(`Bubble '${name}' not found in factory registry`);
    }
    // Always pass params, even if undefined
    return new BubbleClass(params, context) as unknown as T;
  }

  getDetailedDependencies(name: BubbleName): BubbleDependencySpec[] {
    return this.detailedDeps.get(name) || [];
  }

  /**
   * List all registered bubble names
   */
  list(): BubbleName[] {
    return Array.from(this.registry.keys());
  }

  // Return a list of bubbles to be used in the BubbleFlow code generator
  listBubblesForCodeGenerator(): BubbleName[] {
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
      'app-rankings-tool',
      'youtube-tool',
      'github',
      'eleven-labs',
      'followupboss',
      'agi-inc',
      'airtable',
      'notion',
      'insforge-db',
      'ragbits-ingest',
      'ragbits-search',
      'ragbits-index',
      'ragbits-generation',
      'crewai-orchestration',
      'crewai-research',
      'ace-tools',
      'workflow-orchestrator',
      'qdrant',
      'elasticsearch',
      'redis',
      'sendgrid',
      'twilio',
      'stripe',
      'webhook',
      'crewai',
      'airtable-wrapper',
      'openevolve-slack',
      'openevolve-gmail',
      'openevolve-http',
      'openevolve-github',
      'openevolve-apify',
      'openevolve-google-drive',
      'openevolve-google-sheets',
      'openevolve-airtable',
      'openevolve-notion',
      'openevolve-postgresql',
      'openevolve-workflow',
      'openevolve-execution',
      'openevolve-team',
      'openevolve-gauntlet',
      'openevolve-settings',
      'openevolve-icr',
      'openevolve-determinism',
      'openevolve-decomposition',
      'openevolve-decomposition-workflow',
      'openevolve-knowledge-engine',
      'openevolve-workflow-orchestrator',
      'openevolve-ace-tools',
      'openevolve-crewai',
      'openevolve-leanaide',
      'openevolve-z3prover',
      'openevolve-gauntlet-testing',
      'log-parser-tool',
      'metrics-collector-tool',
      'vector-search-tool',
      'csv-processor-tool',
      'json-validator-tool',
      'data-transformer-tool',
      'file-processor-tool',
      'image-processor-tool',
      'xml-parser-tool',
      'pdf-generator-tool',
      'email-validator-tool',
      'url-validator-tool',
      'code-formatter-tool',
      'text-analyzer-tool',
      'data-enrichment-workflow',
      'backup-restore-workflow',
      'monitoring-alert-workflow',
      'etl-pipeline-workflow',
      'api-aggregator-workflow',
      'scheduled-task-workflow',
      'event-handler-workflow',
      'multi-step-approval-workflow',
      'webhook-repeater-workflow',
      'openevolve-oneke',
      'openevolve-gket',
      'openevolve-evolution-trigger',
      'openevolve-evolution-application',
      'openevolve-evolution-validation',
      'openevolve-metrics-collector',
      'openevolve-knowledge-retrieval',
      'openevolve-knowledge-capture',
      'openevolve-evolution-pipeline',
      'openevolve-continuous-evolution',
      'openevolve-adaptive-evolution',
    ] as BubbleName[];
  }

  /**
   * Get the class names (e.g., 'SlackBubble', 'PostgreSQLBubble') for all bubbles
   * available for code generation. Used to generate import statements.
   */
  listBubbleClassNamesForCodeGenerator(): string[] {
    const bubbleNames = this.listBubblesForCodeGenerator();
    const classNames: string[] = [];

    for (const name of bubbleNames) {
      const bubbleClass = this.registry.get(name);
      if (bubbleClass && bubbleClass.name) {
        classNames.push(bubbleClass.name);
      }
    }

    return classNames;
  }

  /**
   * Get a mapping of bubble names to class names for code generation.
   * Returns object like { 'slack': 'SlackBubble', 'postgresql': 'PostgreSQLBubble' }
   */
  getBubbleNameToClassNameMap(): Record<string, string> {
    const bubbleNames = this.listBubblesForCodeGenerator();
    const mapping: Record<string, string> = {};

    for (const name of bubbleNames) {
      const bubbleClass = this.registry.get(name);
      if (bubbleClass && bubbleClass.name) {
        mapping[name] = bubbleClass.name;
      }
    }

    return mapping;
  }

  async registerDefaults(): Promise<void> {
    // Import and register all default bubbles
    // This will be implemented in a separate file to avoid circular deps
    // Register all default bubbles

    const { HelloWorldBubble } = (await this.safeImport(
      './bubbles/service-bubble/hello-world.js'
    )) ?? {};
    const { AIAgentBubble } = (await this.safeImport(
      './bubbles/service-bubble/ai-agent.js'
    )) ?? {};
    const { PostgreSQLBubble } = (await this.safeImport(
      './bubbles/service-bubble/postgresql.js'
    )) ?? {};
    const { SlackBubble } = (await this.safeImport('./bubbles/service-bubble/slack.js')) ?? {};
    const { TelegramBubble } = (await this.safeImport(
      './bubbles/service-bubble/telegram.js'
    )) ?? {};
    const { ResendBubble } = (await this.safeImport('./bubbles/service-bubble/resend.js')) ?? {};
    const { HttpBubble } = (await this.safeImport('./bubbles/service-bubble/http.js')) ?? {};
    const { StorageBubble } = (await this.safeImport(
      './bubbles/service-bubble/storage.js'
    )) ?? {};
    const { GoogleDriveBubble } = (await this.safeImport(
      './bubbles/service-bubble/google-drive.js'
    )) ?? {};
    const { GmailBubble } = (await this.safeImport('./bubbles/service-bubble/gmail.js')) ?? {};
    const { GoogleSheetsBubble } = (await this.safeImport(
      './bubbles/service-bubble/google-sheets'
    )) ?? {};
    const { GoogleCalendarBubble } = (await this.safeImport(
      './bubbles/service-bubble/google-calendar.js'
    )) ?? {};
    const { ApifyBubble } = (await this.safeImport('./bubbles/service-bubble/apify')) ?? {};
    const { GithubBubble } = (await this.safeImport('./bubbles/service-bubble/github.js')) ?? {};
    const { FollowUpBossBubble } = (await this.safeImport(
      './bubbles/service-bubble/followupboss.js'
    )) ?? {};
    const { NotionBubble } = (await this.safeImport(
      './bubbles/service-bubble/notion/notion.js'
    )) ?? {};
    const { DatabaseAnalyzerWorkflowBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/database-analyzer.workflow.js'
    )) ?? {};
    const { SlackNotifierWorkflowBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/slack-notifier.workflow.js'
    )) ?? {};
    const { SlackDataAssistantWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/slack-data-assistant.workflow.js'
    )) ?? {};

    const { ListBubblesTool } = (await this.safeImport(
      './bubbles/tool-bubble/list-bubbles-tool.js'
    )) ?? {};
    const { GetBubbleDetailsTool } = (await this.safeImport(
      './bubbles/tool-bubble/get-bubble-details-tool.js'
    )) ?? {};
    const { GetTriggerDetailTool } = (await this.safeImport(
      './bubbles/tool-bubble/get-trigger-detail-tool.js'
    )) ?? {};
    const { ListCapabilitiesTool } = (await this.safeImport(
      './bubbles/tool-bubble/list-capabilities-tool.js'
    )) ?? {};
    const { AppRankingsTool } = (await this.safeImport(
      './bubbles/tool-bubble/app-rankings-tool.js'
    )) ?? {};
    const { PeopleSearchTool } = (await this.safeImport(
      './bubbles/tool-bubble/people-search-tool.js'
    )) ?? {};
    const { SQLQueryTool } = (await this.safeImport(
      './bubbles/tool-bubble/sql-query-tool.js'
    )) ?? {};
    const { ChartJSTool } = (await this.safeImport(
      './bubbles/tool-bubble/chart-js-tool.js'
    )) ?? {};
    const { BubbleFlowValidationTool } = (await this.safeImport(
      './bubbles/tool-bubble/bubbleflow-validation-tool.js'
    )) ?? {};
    const { EditBubbleFlowTool } = (await this.safeImport(
      './bubbles/tool-bubble/code-edit-tool.js'
    )) ?? {};
    const { WebSearchTool } = (await this.safeImport(
      './bubbles/tool-bubble/web-search-tool.js'
    )) ?? {};
    const { WebScrapeTool } = (await this.safeImport(
      './bubbles/tool-bubble/web-scrape-tool.js'
    )) ?? {};
    const { WebExtractTool } = (await this.safeImport(
      './bubbles/tool-bubble/web-extract-tool.js'
    )) ?? {};
    const { ResearchAgentTool } = (await this.safeImport(
      './bubbles/tool-bubble/research-agent-tool.js'
    )) ?? {};
    const { RedditScrapeTool } = (await this.safeImport(
      './bubbles/tool-bubble/reddit-scrape-tool.js'
    )) ?? {};
    const { InstagramTool } = (await this.safeImport(
      './bubbles/tool-bubble/instagram-tool.js'
    )) ?? {};
    const { LinkedInTool } = (await this.safeImport(
      './bubbles/tool-bubble/linkedin-tool.js'
    )) ?? {};
    const { YouTubeTool } = (await this.safeImport(
      './bubbles/tool-bubble/youtube-tool.js'
    )) ?? {};
    const { TikTokTool } = (await this.safeImport('./bubbles/tool-bubble/tiktok-tool.js')) ?? {};
    const { TwitterTool } = (await this.safeImport(
      './bubbles/tool-bubble/twitter-tool.js'
    )) ?? {};
    const { GoogleMapsTool } = (await this.safeImport(
      './bubbles/tool-bubble/google-maps-tool.js'
    )) ?? {};
    const { LogParserTool } = (await this.safeImport(
      './bubbles/tool-bubble/log-parser-tool.js'
    )) ?? {};
    const { MetricsCollectorTool } = (await this.safeImport(
      './bubbles/tool-bubble/metrics-collector-tool.js'
    )) ?? {};
    const { VectorSearchTool } = (await this.safeImport(
      './bubbles/tool-bubble/vector-search-tool.js'
    )) ?? {};
    const { CSVProcessorTool } = (await this.safeImport(
      './bubbles/tool-bubble/csv-processor-tool.js'
    )) ?? {};
    const { JSONValidatorTool } = (await this.safeImport(
      './bubbles/tool-bubble/json-validator-tool.js'
    )) ?? {};
    const { DataTransformerTool } = (await this.safeImport(
      './bubbles/tool-bubble/data-transformer-tool.js'
    )) ?? {};
    const { FileProcessorTool } = (await this.safeImport(
      './bubbles/tool-bubble/file-processor-tool.js'
    )) ?? {};
    const { ImageProcessorTool } = (await this.safeImport(
      './bubbles/tool-bubble/image-processor-tool.js'
    )) ?? {};
    const { XMLParserTool } = (await this.safeImport(
      './bubbles/tool-bubble/xml-parser-tool.js'
    )) ?? {};
    const { PDFGeneratorTool } = (await this.safeImport(
      './bubbles/tool-bubble/pdf-generator-tool.js'
    )) ?? {};
    const { EmailValidatorTool } = (await this.safeImport(
      './bubbles/tool-bubble/email-validator-tool.js'
    )) ?? {};
    const { URLValidatorTool } = (await this.safeImport(
      './bubbles/tool-bubble/url-validator-tool.js'
    )) ?? {};
    const { CodeFormatterTool } = (await this.safeImport(
      './bubbles/tool-bubble/code-formatter-tool.js'
    )) ?? {};
    const { TextAnalyzerTool } = (await this.safeImport(
      './bubbles/tool-bubble/text-analyzer-tool.js'
    )) ?? {};
    const { SlackFormatterAgentBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/slack-formatter-agent.js'
    )) ?? {};
    const { PDFFormOperationsWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/pdf-form-operations.workflow.js'
    )) ?? {};
    const { PDFOcrWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/pdf-ocr.workflow.js'
    )) ?? {};
    const { GenerateDocumentWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/generate-document.workflow.js'
    )) ?? {};
    const { ParseDocumentWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/parse-document.workflow.js'
    )) ?? {};
    const { DataEnrichmentWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/data-enrichment.workflow.js'
    )) ?? {};
    const { BackupRestoreWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/backup-restore.workflow.js'
    )) ?? {};
    const { MonitoringAlertWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/monitoring-alert.workflow.js'
    )) ?? {};
    const { ETLPipelineWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/etl-pipeline.workflow.js'
    )) ?? {};
    const { APIAggregatorWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/api-aggregator.workflow.js'
    )) ?? {};
    const { ScheduledTaskWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/scheduled-task.workflow.js'
    )) ?? {};
    const { EventHandlerWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/event-handler.workflow.js'
    )) ?? {};
    const { MultiStepApprovalWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/multi-step-approval.workflow.js'
    )) ?? {};
    const { WebhookRepeaterWorkflow } = (await this.safeImport(
      './bubbles/workflow-bubble/webhook-repeater.workflow.js'
    )) ?? {};
    const { ElevenLabsBubble } = (await this.safeImport(
      './bubbles/service-bubble/eleven-labs.js'
    )) ?? {};
    const { AGIIncBubble } = (await this.safeImport(
      './bubbles/service-bubble/agi-inc.js'
    )) ?? {};
    const { AirtableBubble } = (await this.safeImport(
      './bubbles/service-bubble/airtable.js'
    )) ?? {};
    const { FirecrawlBubble } = (await this.safeImport(
      './bubbles/service-bubble/firecrawl.js'
    )) ?? {};
    const { InsForgeDbBubble } = (await this.safeImport(
      './bubbles/service-bubble/insforge-db.js'
    )) ?? {};
    const { AceToolsBubble } = (await this.safeImport(
      './bubbles/service-bubble/ace-tools-bubble.js'
    )) ?? {};
    const { WorkflowOrchestratorBubble } = (await this.safeImport(
      './bubbles/service-bubble/workflow-orchestrator-bubble.js'
    )) ?? {};
    const { QdrantBubble } = (await this.safeImport(
      './bubbles/service-bubble/qdrant-bubble.js'
    )) ?? {};
    const { ElasticsearchBubble } = (await this.safeImport(
      './bubbles/service-bubble/elasticsearch-bubble.js'
    )) ?? {};
    const { RedisBubble } = (await this.safeImport(
      './bubbles/service-bubble/redis-bubble.js'
    )) ?? {};
    const { SendGridBubble } = (await this.safeImport(
      './bubbles/service-bubble/sendgrid-bubble.js'
    )) ?? {};
    const { TwilioBubble } = (await this.safeImport(
      './bubbles/service-bubble/twilio-bubble.js'
    )) ?? {};
    const { StripeBubble } = (await this.safeImport(
      './bubbles/service-bubble/stripe-bubble.js'
    )) ?? {};
    const { WebhookBubble } = (await this.safeImport(
      './bubbles/service-bubble/webhook-bubble.js'
    )) ?? {};
    const { AirtableWrapperBubble } = (await this.safeImport(
      './bubbles/service-bubble/airtable-wrapper.js'
    )) ?? {};
    const { OpenEvolveWorkflowBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-workflow-bubble.js'
    )) ?? {};
    const { OpenEvolveExecutionBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-execution-bubble.js'
    )) ?? {};
    const { OpenEvolveTeamBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-team-bubble.js'
    )) ?? {};
    const { OpenEvolveGauntletBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-gauntlet-bubble.js'
    )) ?? {};
    const { OpenEvolveSettingsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-settings-bubble.js'
    )) ?? {};
    const { OpenEvolveIcrBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-icr-bubble.js'
    )) ?? {};
    const { OpenEvolveDeterminismBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-determinism-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionWorkflowBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-workflow-bubble.js'
    )) ?? {};
    const { OpenEvolveTeamMembersBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-team-members-bubble.js'
    )) ?? {};
    const { OpenEvolveTeamAssignBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-team-assign-bubble.js'
    )) ?? {};
    const { OpenEvolveTeamTemplatesBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-team-templates-bubble.js'
    )) ?? {};
    const { OpenEvolveTeamLlmsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-team-llms-bubble.js'
    )) ?? {};
    const { OpenEvolveTeamCredentialsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-team-credentials-bubble.js'
    )) ?? {};
    const { OpenEvolveVersionControlBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-version-control-bubble.js'
    )) ?? {};
    const { OpenEvolveValidationBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-validation-bubble.js'
    )) ?? {};
    const { OpenEvolveParametersBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-parameters-bubble.js'
    )) ?? {};
    const { OpenEvolveProvidersBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-providers-bubble.js'
    )) ?? {};
    const { OpenEvolveAuditLogsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-audit-logs-bubble.js'
    )) ?? {};
    const { OpenEvolveAutoApprovalBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-auto-approval-bubble.js'
    )) ?? {};
    const { OpenEvolvePromptsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-prompts-bubble.js'
    )) ?? {};
    const { OpenEvolveContentTemplatesBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-content-templates-bubble.js'
    )) ?? {};
    const { OpenEvolveContentValidateBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-content-validate-bubble.js'
    )) ?? {};
    const { OpenEvolveSecurityBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-security-bubble.js'
    )) ?? {};
    const { OpenEvolveGatewayBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-gateway-bubble.js'
    )) ?? {};
    const { OpenEvolveV1EvolveBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-v1-evolve-bubble.js'
    )) ?? {};
    const { OpenEvolveV1RunsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-v1-runs-bubble.js'
    )) ?? {};
    const { OpenEvolveV1WorkflowLifecycleBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-v1-workflow-lifecycle-bubble.js'
    )) ?? {};
    const { OpenEvolveIntegratedRunBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-integrated-run-bubble.js'
    )) ?? {};
    const { OpenEvolveOrchestrationModelsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-orchestration-models-bubble.js'
    )) ?? {};
    const { OpenEvolveOrchestrationEnsembleBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-orchestration-ensemble-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeEngineBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-engine-bubble.js'
    )) ?? {};
    const { OpenEvolveWorkflowOrchestratorBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-workflow-orchestrator-bubble.js'
    )) ?? {};
    const { OpenEvolveAceToolsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-ace-tools-bubble.js'
    )) ?? {};
    const { OpenEvolveCrewAIBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-crewai-bubble.js'
    )) ?? {};
    const { OpenEvolveLeanAideBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-leanaide-bubble.js'
    )) ?? {};
    const { OpenEvolveZ3ProverBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-z3prover-bubble.js'
    )) ?? {};
    const { OpenEvolveOneKEBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-oneke-bubble.js'
    )) ?? {};
    const { OpenEvolveGKETBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-gket-bubble.js'
    )) ?? {};
    const { OpenEvolveEvolutionTriggerBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-evolution-trigger-bubble.js'
    )) ?? {};
    const { OpenEvolveEvolutionApplicationBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-evolution-application-bubble.js'
    )) ?? {};
    const { OpenEvolveEvolutionValidationBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-evolution-validation-bubble.js'
    )) ?? {};
    const { OpenEvolveMetricsCollectorBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-metrics-collector-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeRetrievalBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-retrieval-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeCaptureBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-capture-bubble.js'
    )) ?? {};
    const { OpenEvolveEvolutionPipelineBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-evolution-pipeline-bubble.js'
    )) ?? {};
    const { OpenEvolveContinuousEvolutionBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-continuous-evolution-bubble.js'
    )) ?? {};
    const { OpenEvolveAdaptiveEvolutionBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-adaptive-evolution-bubble.js'
    )) ?? {};
    const { OpenEvolveGauntletTestingBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-gauntlet-testing-bubble.js'
    )) ?? {};
    const { SlackBubble: OpenEvolveSlackBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/slack-bubble.js'
    )) ?? {};
    const { GmailBubble: OpenEvolveGmailBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/gmail-bubble.js'
    )) ?? {};
    const { HttpBubble: OpenEvolveHttpBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/http-bubble.js'
    )) ?? {};
    const { GithubBubble: OpenEvolveGithubBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/github-bubble.js'
    )) ?? {};
    const { ApifyBubble: OpenEvolveApifyBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/apify-bubble.js'
    )) ?? {};
    const { GoogleDriveBubble: OpenEvolveGoogleDriveBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/google-drive-bubble.js'
    )) ?? {};
    const { GoogleSheetsBubble: OpenEvolveGoogleSheetsBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/google-sheets-bubble.js'
    )) ?? {};
    const { AirtableBubble: OpenEvolveAirtableBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/airtable-bubble.js'
    )) ?? {};
    const { NotionBubble: OpenEvolveNotionBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/notion-bubble.js'
    )) ?? {};
    const { PostgreSQLBubble: OpenEvolvePostgreSQLBubbleBase } = (await this.safeImport(
      './bubbles/service-bubble/postgresql-bubble.js'
    )) ?? {};
    // Import RAGBits bubbles
    const { RAGBitsIngestBubble } = (await this.safeImport(
      '../../ragbits-bubblelab-integration/bubbles/ingest/RAGBitsIngestBubble.ts'
    )) ?? {};
    const { RAGBitsSearchBubble } = (await this.safeImport(
      '../../ragbits-bubblelab-integration/bubbles/search/RAGBitsSearchBubble.ts'
    )) ?? {};
    const { RAGBitsIndexBubble } = (await this.safeImport(
      '../../ragbits-bubblelab-integration/bubbles/index/RAGBitsIndexBubble.ts'
    )) ?? {};
    const { RAGBitsGenerationBubble } = (await this.safeImport(
      '../../ragbits-bubblelab-integration/bubbles/generation/RAGBitsGenerationBubble.ts'
    )) ?? {};

    // Import CrewAI bubbles
    const { CrewAIOrchestrationBubble, CrewAIResearchBubble } = (await this.safeImport(
      '../../ragbits-bubblelab-integration/bubbles/crewai/CrewAIOrchestrationBubble.ts'
    )) ?? {};

    const wrapBubbleName = <T extends BubbleClassWithMetadata<any>>(
      BubbleClass: T | undefined,
      bubbleName: BubbleName
    ) => {
      if (!BubbleClass) {
        return undefined as unknown as T;
      }
      return class extends (BubbleClass as unknown as new (...args: any[]) => any) {
        static readonly bubbleName = bubbleName;
      } as unknown as T;
    };
    const CrewAIBubbleAlias = wrapBubbleName(
      OpenEvolveCrewAIBubble as BubbleClassWithMetadata,
      'crewai' as BubbleName
    );
    const { BrowserBaseBubble } = await import(
      './bubbles/service-bubble/browserbase/index.js'
    );
    const { AmazonShoppingTool } = await import(
      './bubbles/tool-bubble/amazon-shopping-tool/index.js'
    );
    const { CrustdataBubble } = await import(
      './bubbles/service-bubble/crustdata/index.js'
    );
    const { CompanyEnrichmentTool } = await import(
      './bubbles/tool-bubble/company-enrichment-tool.js'
    );
    const { JiraBubble } = await import(
      './bubbles/service-bubble/jira/index.js'
    );
    const { ConfluenceBubble } = await import(
      './bubbles/service-bubble/confluence/index.js'
    );
    const { AshbyBubble } = await import(
      './bubbles/service-bubble/ashby/index.js'
    );
    const { FullEnrichBubble } = await import(
      './bubbles/service-bubble/fullenrich/index.js'
    );
    const {
      LinkedInConnectionTool,
      LinkedInSentInvitationsTool,
      LinkedInReceivedInvitationsTool,
      LinkedInAcceptInvitationsTool,
    } = await import('./bubbles/tool-bubble/browser-tools/index.js');
    const { SendSafelyBubble } = await import(
      './bubbles/service-bubble/sendsafely/index.js'
    );
    const { YCScraperTool } = await import(
      './bubbles/tool-bubble/yc-scraper-tool.js'
    );
    const { PosthogBubble } = await import(
      './bubbles/service-bubble/posthog/index.js'
    );
    const { LinearBubble } = await import(
      './bubbles/service-bubble/linear/index.js'
    );
    const { AttioBubble } = await import(
      './bubbles/service-bubble/attio/index.js'
    );
    const { HubSpotBubble } = await import(
      './bubbles/service-bubble/hubspot/index.js'
    );
    const { S3Bubble } = await import('./bubbles/service-bubble/s3/index.js');
    const { AssembledBubble } = await import(
      './bubbles/service-bubble/assembled/index.js'
    );
    const { XeroBubble } = await import(
      './bubbles/service-bubble/xero/index.js'
    );
    const { RampBubble } = await import(
      './bubbles/service-bubble/ramp/index.js'
    );
    const { ZendeskBubble } = await import(
      './bubbles/service-bubble/zendesk/index.js'
    );
    const { SlabBubble } = await import(
      './bubbles/service-bubble/slab/index.js'
    );
    const { SnowflakeBubble } = await import(
      './bubbles/service-bubble/snowflake/index.js'
    );
    const { SalesforceBubble } = await import(
      './bubbles/service-bubble/salesforce/index.js'
    );
    const { AsanaBubble } = await import(
      './bubbles/service-bubble/asana/index.js'
    );
    const { DiscordBubble } = await import(
      './bubbles/service-bubble/discord/index.js'
    );
    const { SortlyBubble } = await import(
      './bubbles/service-bubble/sortly/index.js'
    );
    const { DocuSignBubble } = await import(
      './bubbles/service-bubble/docusign/index.js'
    );
    const { MetabaseBubble } = await import(
      './bubbles/service-bubble/metabase/index.js'
    );
    const { ClerkBubble } = await import(
      './bubbles/service-bubble/clerk/index.js'
    );
    const { GranolaBubble } = await import(
      './bubbles/service-bubble/granola/index.js'
    );
    const { MemberfulBubble } = await import(
      './bubbles/service-bubble/memberful/index.js'
    );
    const { LumaBubble } = await import('./bubbles/service-bubble/luma.js');
    const { ZoomBubble } = await import(
      './bubbles/service-bubble/zoom/index.js'
    );

    const { OpenEvolveDecompositionPlanGetBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-plan-get-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionPlanUpdateBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-plan-update-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionExecuteBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-execute-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionExecutionStatusBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-execution-status-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionSettingsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-settings-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionResultsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-results-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionTelemetryBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-telemetry-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionTruthPackageBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-truth-package-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionResourceUsageBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-decomposition-resource-usage-bubble.js'
    )) ?? {};
    const { OpenEvolveGauntletExecuteBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-gauntlet-execute-bubble.js'
    )) ?? {};
    const { OpenEvolveGauntletExecutionStatusBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-gauntlet-execution-status-bubble.js'
    )) ?? {};
    const { OpenEvolveGauntletExecutionsListBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-gauntlet-executions-list-bubble.js'
    )) ?? {};
    const { OpenEvolveAdversarialRunBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-adversarial-run-bubble.js'
    )) ?? {};
    const { OpenEvolveEvolutionRunBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-evolution-run-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeGraphBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-graph-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeStatsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-stats-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeRecommendationsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-recommendations-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeExportBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-export-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeImportBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-import-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeGetBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-get-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeDeleteBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-delete-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeDocumentsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-documents-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeEmbedBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-embed-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeSyncBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-sync-bubble.js'
    )) ?? {};
    const { OpenEvolveMonitoringDashboardBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-monitoring-dashboard-bubble.js'
    )) ?? {};
    const { OpenEvolveMonitoringAlertsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-monitoring-alerts-bubble.js'
    )) ?? {};
    const { OpenEvolveMonitoringServicesBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-monitoring-services-bubble.js'
    )) ?? {};
    const { OpenEvolveMonitoringLogsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-monitoring-logs-bubble.js'
    )) ?? {};
    const { OpenEvolveMonitoringMetricsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-monitoring-metrics-bubble.js'
    )) ?? {};
    const { OpenEvolveMonitoringHealthBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-monitoring-health-bubble.js'
    )) ?? {};
    const { OpenEvolveAnalyticsStatisticsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-analytics-statistics-bubble.js'
    )) ?? {};
    const { OpenEvolveAnalyticsPerformanceBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-analytics-performance-bubble.js'
    )) ?? {};
    const { OpenEvolveAnalyticsKnowledgeBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-analytics-knowledge-bubble.js'
    )) ?? {};
    const { OpenEvolveAnalyticsWorkflowBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-analytics-workflow-bubble.js'
    )) ?? {};
    const { OpenEvolveSettingsIcrBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-settings-icr-bubble.js'
    )) ?? {};
    const { OpenEvolveSettingsDeterminismBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-settings-determinism-bubble.js'
    )) ?? {};
    const { OpenEvolveSettingsAdaptiveDecompositionBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-settings-adaptive-decomposition-bubble.js'
    )) ?? {};
    const { OpenEvolveSettingsMdapMakerBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-settings-mdap-maker-bubble.js'
    )) ?? {};
    const { OpenEvolveSettingsRomaMdapMakerBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-settings-roma-mdap-maker-bubble.js'
    )) ?? {};
    const { OpenEvolveDecompositionToEvolutionBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-decomposition-to-evolution-bubble.js'
    )) ?? {};
    const { OpenEvolveGauntletRedBlueGoldBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-gauntlet-red-blue-gold-bubble.js'
    )) ?? {};
    const { OpenEvolveAdversarialToEvolutionBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-adversarial-to-evolution-bubble.js'
    )) ?? {};
    const { OpenEvolveFullEvolutionLifecycleBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-full-evolution-lifecycle-bubble.js'
    )) ?? {};
    const { OpenEvolveWorkflowDecompositionPipelineBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-workflow-decomposition-pipeline-bubble.js'
    )) ?? {};
    const { OpenEvolveVerifiedDeploymentBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-verified-deployment-bubble.js'
    )) ?? {};

    const { OpenEvolveBubblelabsStatusBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-status-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsInitBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-init-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsWorkflowDefinitionsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-workflow-definitions-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsWorkflowInstancesBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-workflow-instances-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsAceSkillbookBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-ace-skillbook-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsAcePatternsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-ace-patterns-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsZ3Bubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-z3-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsRomaBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-roma-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsKnowledgeStoreBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-knowledge-store-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsKnowledgeQueryBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-knowledge-query-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsAnalyticsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-analytics-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsLeanaideProveBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-leanaide-prove-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsIntegrationsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-integrations-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsControlCatalogBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-control-catalog-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsControlDiscoverBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-control-discover-bubble.js'
    )) ?? {};
    const { OpenEvolveBubblelabsControlExecuteBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-bubblelabs-control-execute-bubble.js'
    )) ?? {};
    const { OpenEvolveIcrOverviewBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-icr-overview-bubble.js'
    )) ?? {};
    const { OpenEvolveIcrComponentsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-icr-components-bubble.js'
    )) ?? {};
    const { OpenEvolveIcrRefinementsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-icr-refinements-bubble.js'
    )) ?? {};
    const { OpenEvolveIcrVlmBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-icr-vlm-bubble.js'
    )) ?? {};
    const { OpenEvolveIcrAnalyticsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-icr-analytics-bubble.js'
    )) ?? {};
    const { OpenEvolveIcrConfigBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-icr-config-bubble.js'
    )) ?? {};
    const { OpenEvolveIcrDashboardBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-icr-dashboard-bubble.js'
    )) ?? {};
    const { OpenEvolveAdaptiveMdapDashboardBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-adaptive-mdap-dashboard-bubble.js'
    )) ?? {};
    const { OpenEvolveAdaptiveMdapProfilesBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-adaptive-mdap-profiles-bubble.js'
    )) ?? {};
    const { OpenEvolveAdaptiveMdapCostBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-adaptive-mdap-cost-bubble.js'
    )) ?? {};
    const { OpenEvolveAdaptiveMdapComplexityBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-adaptive-mdap-complexity-bubble.js'
    )) ?? {};
    const { OpenEvolveAdaptiveMdapAllocateBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-adaptive-mdap-allocate-bubble.js'
    )) ?? {};
    const { OpenEvolveRagbitsSearchBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-ragbits-search-bubble.js'
    )) ?? {};
    const { OpenEvolveRagbitsIngestBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-ragbits-ingest-bubble.js'
    )) ?? {};
    const { OpenEvolveRagbitsStatsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-ragbits-stats-bubble.js'
    )) ?? {};
    const { OpenEvolveDspyAssessBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-dspy-assess-bubble.js'
    )) ?? {};
    const { OpenEvolveDspyFixBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-dspy-fix-bubble.js'
    )) ?? {};
    const { OpenEvolvePygraphistryBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-pygraphistry-bubble.js'
    )) ?? {};
    const { OpenEvolveWeb3StatusBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-web3-status-bubble.js'
    )) ?? {};
    const { OpenEvolveWeb3IngestBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-web3-ingest-bubble.js'
    )) ?? {};
    const { OpenEvolveWeb3SlitherBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-web3-slither-bubble.js'
    )) ?? {};
    const { OpenEvolveWeb3FoundryBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-web3-foundry-bubble.js'
    )) ?? {};
    const { OpenEvolveWeb3InvariantsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-web3-invariants-bubble.js'
    )) ?? {};
    const { OpenEvolveWeb3ExploitWitnessBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-web3-exploit-witness-bubble.js'
    )) ?? {};
    const { OpenEvolveWeb3AuditExploitBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-web3-audit-exploit-bubble.js'
    )) ?? {};
    const { OpenEvolveWeb3McpInventoryBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-web3-mcp-inventory-bubble.js'
    )) ?? {};
    const { OpenEvolveSovereignStatusBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-sovereign-status-bubble.js'
    )) ?? {};
    const { OpenEvolveSovereignProblemsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-sovereign-problems-bubble.js'
    )) ?? {};
    const { OpenEvolveSovereignPlansBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-sovereign-plans-bubble.js'
    )) ?? {};
    const { OpenEvolveSovereignStatsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-sovereign-stats-bubble.js'
    )) ?? {};
    const { OpenEvolveSovereignRunBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-sovereign-run-bubble.js'
    )) ?? {};
    const { OpenEvolveSovereignRunsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-sovereign-runs-bubble.js'
    )) ?? {};
    const { OpenEvolveSovereignPipelineBubble } = (await this.safeImport(
      './bubbles/workflow-bubble/openevolve-sovereign-pipeline-bubble.js'
    )) ?? {};
    const { OpenEvolveMakerStatusBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-maker-status-bubble.js'
    )) ?? {};
    const { OpenEvolveMakerToolsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-maker-tools-bubble.js'
    )) ?? {};
    const { OpenEvolveMakerTestBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-maker-test-bubble.js'
    )) ?? {};
    const { OpenEvolveMakerValidateBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-maker-validate-bubble.js'
    )) ?? {};
    const { OpenEvolveMakerExecuteBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-maker-execute-bubble.js'
    )) ?? {};
    const { OpenEvolveMakerDelegationsBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-maker-delegations-bubble.js'
    )) ?? {};
    const { OpenEvolveMakerDelegationsSyncBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-maker-delegations-sync-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeExplorerStatusBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-explorer-status-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeExplorerQueryBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-explorer-query-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeExplorerHistoryBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-explorer-history-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeExplorerExtractBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-explorer-extract-bubble.js'
    )) ?? {};
    const { OpenEvolveKnowledgeExplorerExtractFileBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-knowledge-explorer-extract-file-bubble.js'
    )) ?? {};
    const { OpenEvolveSuggestionsContentBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-suggestions-content-bubble.js'
    )) ?? {};
    const { OpenEvolveSuggestionsClassificationBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-suggestions-classification-bubble.js'
    )) ?? {};
    const { OpenEvolveSuggestionsSecurityBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-suggestions-security-bubble.js'
    )) ?? {};
    const { OpenEvolveSuggestionsImprovementBubble } = (await this.safeImport(
      './bubbles/service-bubble/openevolve-suggestions-improvement-bubble.js'
    )) ?? {};

    // Create the default factory instance
    this.register('hello-world', HelloWorldBubble as BubbleClassWithMetadata);
    this.register('ai-agent', AIAgentBubble as BubbleClassWithMetadata);
    this.register('postgresql', PostgreSQLBubble as BubbleClassWithMetadata);
    this.register('slack', SlackBubble as BubbleClassWithMetadata);
    this.register(
      'telegram' as BubbleName,
      TelegramBubble as unknown as BubbleClassWithMetadata
    );
    this.register('resend', ResendBubble as BubbleClassWithMetadata);
    this.register('http', HttpBubble as BubbleClassWithMetadata);
    this.register('storage', StorageBubble as BubbleClassWithMetadata);
    this.register('google-drive', GoogleDriveBubble as BubbleClassWithMetadata);
    this.register('gmail', GmailBubble as BubbleClassWithMetadata);
    this.register(
      'google-sheets',
      GoogleSheetsBubble as BubbleClassWithMetadata
    );
    this.register(
      'google-calendar',
      GoogleCalendarBubble as BubbleClassWithMetadata
    );
    this.register('apify', ApifyBubble as BubbleClassWithMetadata);
    this.register('github', GithubBubble as BubbleClassWithMetadata);
    this.register(
      'followupboss',
      FollowUpBossBubble as BubbleClassWithMetadata
    );
    this.register('notion', NotionBubble as BubbleClassWithMetadata);
    this.register(
      'database-analyzer',
      DatabaseAnalyzerWorkflowBubble as BubbleClassWithMetadata
    );
    this.register(
      'slack-notifier',
      SlackNotifierWorkflowBubble as BubbleClassWithMetadata
    );
    this.register(
      'slack-data-assistant',
      SlackDataAssistantWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'slack-formatter-agent',
      SlackFormatterAgentBubble as BubbleClassWithMetadata
    );
    this.register(
      'pdf-form-operations',
      PDFFormOperationsWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'pdf-ocr-workflow',
      PDFOcrWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'generate-document-workflow',
      GenerateDocumentWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'parse-document-workflow',
      ParseDocumentWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'get-bubble-details-tool',
      GetBubbleDetailsTool as BubbleClassWithMetadata
    );
    this.register(
      'get-trigger-detail-tool',
      GetTriggerDetailTool as BubbleClassWithMetadata
    );
    this.register(
      'list-bubbles-tool',
      ListBubblesTool as BubbleClassWithMetadata
    );
    this.register(
      'list-capabilities-tool',
      ListCapabilitiesTool as BubbleClassWithMetadata
    );
    this.register('sql-query-tool', SQLQueryTool as BubbleClassWithMetadata);
    this.register('chart-js-tool', ChartJSTool as BubbleClassWithMetadata);
    this.register(
      'bubbleflow-validation-tool',
      BubbleFlowValidationTool as BubbleClassWithMetadata
    );
    this.register(
      'code-edit-tool',
      EditBubbleFlowTool as BubbleClassWithMetadata
    );
    this.register('web-search-tool', WebSearchTool as BubbleClassWithMetadata);
    this.register('web-scrape-tool', WebScrapeTool as BubbleClassWithMetadata);
    this.register(
      'web-extract-tool',
      WebExtractTool as BubbleClassWithMetadata
    );
    this.register(
      'research-agent-tool',
      ResearchAgentTool as BubbleClassWithMetadata
    );
    this.register(
      'reddit-scrape-tool',
      RedditScrapeTool as BubbleClassWithMetadata
    );
    this.register('instagram-tool', InstagramTool as BubbleClassWithMetadata);
    this.register('linkedin-tool', LinkedInTool as BubbleClassWithMetadata);
    this.register('tiktok-tool', TikTokTool as BubbleClassWithMetadata);
    this.register('twitter-tool', TwitterTool as BubbleClassWithMetadata);
    this.register(
      'google-maps-tool',
      GoogleMapsTool as BubbleClassWithMetadata
    );
    this.register(
      'app-rankings-tool',
      AppRankingsTool as BubbleClassWithMetadata
    );
    this.register('youtube-tool', YouTubeTool as BubbleClassWithMetadata);
    this.register('web-crawl-tool', WebCrawlTool as BubbleClassWithMetadata);
    this.register('eleven-labs', ElevenLabsBubble as BubbleClassWithMetadata);
    this.register('agi-inc', AGIIncBubble as BubbleClassWithMetadata);
    this.register('airtable', AirtableBubble as BubbleClassWithMetadata);
    this.register('firecrawl', FirecrawlBubble as BubbleClassWithMetadata);
    this.register('insforge-db', InsForgeDbBubble as BubbleClassWithMetadata);
    this.register('browserbase', BrowserBaseBubble as BubbleClassWithMetadata);
    this.register(
      'people-search-tool',
      PeopleSearchTool as BubbleClassWithMetadata
    );
    this.register(
      'amazon-shopping-tool',
      AmazonShoppingTool as BubbleClassWithMetadata
    );
    this.register('crustdata', CrustdataBubble as BubbleClassWithMetadata);
    this.register(
      'company-enrichment-tool',
      CompanyEnrichmentTool as BubbleClassWithMetadata
    );
    this.register('jira', JiraBubble as BubbleClassWithMetadata);
    this.register('confluence', ConfluenceBubble as BubbleClassWithMetadata);
    this.register('ashby', AshbyBubble as BubbleClassWithMetadata);
    this.register('fullenrich', FullEnrichBubble as BubbleClassWithMetadata);
    this.register(
      'linkedin-connection-tool',
      LinkedInConnectionTool as unknown as BubbleClassWithMetadata
    );
    this.register(
      'linkedin-sent-invitations-tool',
      LinkedInSentInvitationsTool as unknown as BubbleClassWithMetadata
    );
    this.register(
      'linkedin-received-invitations-tool',
      LinkedInReceivedInvitationsTool as unknown as BubbleClassWithMetadata
    );
    this.register(
      'linkedin-accept-invitations-tool',
      LinkedInAcceptInvitationsTool as unknown as BubbleClassWithMetadata
    );
    this.register('stripe', StripeBubble as BubbleClassWithMetadata);
    this.register('sendsafely', SendSafelyBubble as BubbleClassWithMetadata);
    this.register('yc-scraper-tool', YCScraperTool as BubbleClassWithMetadata);
    this.register('posthog', PosthogBubble as BubbleClassWithMetadata);
    this.register('linear', LinearBubble as BubbleClassWithMetadata);
    this.register('attio', AttioBubble as BubbleClassWithMetadata);
    this.register('hubspot', HubSpotBubble as BubbleClassWithMetadata);
    this.register('s3-storage', S3Bubble as BubbleClassWithMetadata);
    this.register('assembled', AssembledBubble as BubbleClassWithMetadata);
    this.register('xero', XeroBubble as BubbleClassWithMetadata);
    this.register('ramp', RampBubble as BubbleClassWithMetadata);
    this.register('zendesk', ZendeskBubble as BubbleClassWithMetadata);
    this.register('slab', SlabBubble as BubbleClassWithMetadata);
    this.register('snowflake', SnowflakeBubble as BubbleClassWithMetadata);
    this.register('salesforce', SalesforceBubble as BubbleClassWithMetadata);
    this.register('asana', AsanaBubble as BubbleClassWithMetadata);
    this.register('discord', DiscordBubble as BubbleClassWithMetadata);
    this.register('sortly', SortlyBubble as BubbleClassWithMetadata);
    this.register('docusign', DocuSignBubble as BubbleClassWithMetadata);
    this.register('metabase', MetabaseBubble as BubbleClassWithMetadata);
    this.register('clerk', ClerkBubble as BubbleClassWithMetadata);
    this.register('granola', GranolaBubble as BubbleClassWithMetadata);
    this.register('memberful', MemberfulBubble as BubbleClassWithMetadata);
    this.register('luma', LumaBubble as BubbleClassWithMetadata);
    this.register('zoom', ZoomBubble as BubbleClassWithMetadata);

    // Register RAGBits bubbles
    this.register('ragbits-ingest', RAGBitsIngestBubble as BubbleClassWithMetadata);
    this.register('ragbits-search', RAGBitsSearchBubble as BubbleClassWithMetadata);
    this.register('ragbits-index', RAGBitsIndexBubble as BubbleClassWithMetadata);
    this.register('ragbits-generation', RAGBitsGenerationBubble as BubbleClassWithMetadata);

    // Register CrewAI bubbles
    this.register('crewai-orchestration', CrewAIOrchestrationBubble as BubbleClassWithMetadata);
    this.register('crewai-research', CrewAIResearchBubble as BubbleClassWithMetadata);

    // Register OpenEvolve service bubbles (non-conflicting)
    this.register('ace-tools' as BubbleName, AceToolsBubble as BubbleClassWithMetadata);
    this.register(
      'workflow-orchestrator' as BubbleName,
      WorkflowOrchestratorBubble as BubbleClassWithMetadata
    );
    this.register('qdrant' as BubbleName, QdrantBubble as BubbleClassWithMetadata);
    this.register(
      'elasticsearch' as BubbleName,
      ElasticsearchBubble as BubbleClassWithMetadata
    );
    this.register('redis' as BubbleName, RedisBubble as BubbleClassWithMetadata);
    this.register('sendgrid' as BubbleName, SendGridBubble as BubbleClassWithMetadata);
    this.register('twilio' as BubbleName, TwilioBubble as BubbleClassWithMetadata);
    this.register('stripe' as BubbleName, StripeBubble as BubbleClassWithMetadata);
    this.register('webhook' as BubbleName, WebhookBubble as BubbleClassWithMetadata);
    this.register(
      'crewai' as BubbleName,
      CrewAIBubbleAlias as BubbleClassWithMetadata
    );
    this.register(
      'airtable-wrapper' as BubbleName,
      AirtableWrapperBubble as BubbleClassWithMetadata
    );

    // Register OpenEvolve service bubbles with prefixed names to avoid collisions
    const OpenEvolveSlackBubble = wrapBubbleName(
      OpenEvolveSlackBubbleBase as BubbleClassWithMetadata,
      'openevolve-slack' as BubbleName
    );
    const OpenEvolveGmailBubble = wrapBubbleName(
      OpenEvolveGmailBubbleBase as BubbleClassWithMetadata,
      'openevolve-gmail' as BubbleName
    );
    const OpenEvolveHttpBubble = wrapBubbleName(
      OpenEvolveHttpBubbleBase as BubbleClassWithMetadata,
      'openevolve-http' as BubbleName
    );
    const OpenEvolveGithubBubble = wrapBubbleName(
      OpenEvolveGithubBubbleBase as BubbleClassWithMetadata,
      'openevolve-github' as BubbleName
    );
    const OpenEvolveApifyBubble = wrapBubbleName(
      OpenEvolveApifyBubbleBase as BubbleClassWithMetadata,
      'openevolve-apify' as BubbleName
    );
    const OpenEvolveGoogleDriveBubble = wrapBubbleName(
      OpenEvolveGoogleDriveBubbleBase as BubbleClassWithMetadata,
      'openevolve-google-drive' as BubbleName
    );
    const OpenEvolveGoogleSheetsBubble = wrapBubbleName(
      OpenEvolveGoogleSheetsBubbleBase as BubbleClassWithMetadata,
      'openevolve-google-sheets' as BubbleName
    );
    const OpenEvolveAirtableBubble = wrapBubbleName(
      OpenEvolveAirtableBubbleBase as BubbleClassWithMetadata,
      'openevolve-airtable' as BubbleName
    );
    const OpenEvolveNotionBubble = wrapBubbleName(
      OpenEvolveNotionBubbleBase as BubbleClassWithMetadata,
      'openevolve-notion' as BubbleName
    );
    const OpenEvolvePostgreSQLBubble = wrapBubbleName(
      OpenEvolvePostgreSQLBubbleBase as BubbleClassWithMetadata,
      'openevolve-postgresql' as BubbleName
    );

    this.register(
      'slack' as BubbleName,
      OpenEvolveSlackBubble as BubbleClassWithMetadata
    );
    this.register(
      'gmail' as BubbleName,
      OpenEvolveGmailBubble as BubbleClassWithMetadata
    );
    this.register(
      'http' as BubbleName,
      OpenEvolveHttpBubble as BubbleClassWithMetadata
    );
    this.register(
      'github' as BubbleName,
      OpenEvolveGithubBubble as BubbleClassWithMetadata
    );
    this.register(
      'apify' as BubbleName,
      OpenEvolveApifyBubble as BubbleClassWithMetadata
    );
    this.register(
      'google-drive' as BubbleName,
      OpenEvolveGoogleDriveBubble as BubbleClassWithMetadata
    );
    this.register(
      'google-sheets' as BubbleName,
      OpenEvolveGoogleSheetsBubble as BubbleClassWithMetadata
    );
    this.register(
      'airtable' as BubbleName,
      OpenEvolveAirtableBubble as BubbleClassWithMetadata
    );
    this.register(
      'notion' as BubbleName,
      OpenEvolveNotionBubble as BubbleClassWithMetadata
    );
    this.register(
      'postgresql' as BubbleName,
      OpenEvolvePostgreSQLBubble as BubbleClassWithMetadata
    );

    // Register OpenEvolve workflow system bubbles
    this.register(
      'openevolve-workflow' as BubbleName,
      OpenEvolveWorkflowBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-execution' as BubbleName,
      OpenEvolveExecutionBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-team' as BubbleName,
      OpenEvolveTeamBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gauntlet' as BubbleName,
      OpenEvolveGauntletBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-settings' as BubbleName,
      OpenEvolveSettingsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-icr' as BubbleName,
      OpenEvolveIcrBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-determinism' as BubbleName,
      OpenEvolveDeterminismBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition' as BubbleName,
      OpenEvolveDecompositionBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-workflow' as BubbleName,
      OpenEvolveDecompositionWorkflowBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-team-members' as BubbleName,
      OpenEvolveTeamMembersBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-team-assign' as BubbleName,
      OpenEvolveTeamAssignBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-team-templates' as BubbleName,
      OpenEvolveTeamTemplatesBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-team-llms' as BubbleName,
      OpenEvolveTeamLlmsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-team-credentials' as BubbleName,
      OpenEvolveTeamCredentialsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-version-control' as BubbleName,
      OpenEvolveVersionControlBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-validation' as BubbleName,
      OpenEvolveValidationBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-parameters' as BubbleName,
      OpenEvolveParametersBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-providers' as BubbleName,
      OpenEvolveProvidersBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-audit-logs' as BubbleName,
      OpenEvolveAuditLogsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-auto-approval' as BubbleName,
      OpenEvolveAutoApprovalBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-prompts' as BubbleName,
      OpenEvolvePromptsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-content-templates' as BubbleName,
      OpenEvolveContentTemplatesBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-content-validate' as BubbleName,
      OpenEvolveContentValidateBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-security' as BubbleName,
      OpenEvolveSecurityBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gateway' as BubbleName,
      OpenEvolveGatewayBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-v1-evolve' as BubbleName,
      OpenEvolveV1EvolveBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-v1-runs' as BubbleName,
      OpenEvolveV1RunsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-v1-workflow-lifecycle' as BubbleName,
      OpenEvolveV1WorkflowLifecycleBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-integrated-run' as BubbleName,
      OpenEvolveIntegratedRunBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-orchestration-models' as BubbleName,
      OpenEvolveOrchestrationModelsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-orchestration-ensemble' as BubbleName,
      OpenEvolveOrchestrationEnsembleBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-plan-get' as BubbleName,
      OpenEvolveDecompositionPlanGetBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-plan-update' as BubbleName,
      OpenEvolveDecompositionPlanUpdateBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-execute' as BubbleName,
      OpenEvolveDecompositionExecuteBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-execution-status' as BubbleName,
      OpenEvolveDecompositionExecutionStatusBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-settings' as BubbleName,
      OpenEvolveDecompositionSettingsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-results' as BubbleName,
      OpenEvolveDecompositionResultsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-telemetry' as BubbleName,
      OpenEvolveDecompositionTelemetryBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-truth-package' as BubbleName,
      OpenEvolveDecompositionTruthPackageBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-resource-usage' as BubbleName,
      OpenEvolveDecompositionResourceUsageBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gauntlet-execute' as BubbleName,
      OpenEvolveGauntletExecuteBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gauntlet-execution-status' as BubbleName,
      OpenEvolveGauntletExecutionStatusBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gauntlet-executions-list' as BubbleName,
      OpenEvolveGauntletExecutionsListBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-adversarial-run' as BubbleName,
      OpenEvolveAdversarialRunBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-evolution-run' as BubbleName,
      OpenEvolveEvolutionRunBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-graph' as BubbleName,
      OpenEvolveKnowledgeGraphBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-stats' as BubbleName,
      OpenEvolveKnowledgeStatsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-recommendations' as BubbleName,
      OpenEvolveKnowledgeRecommendationsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-export' as BubbleName,
      OpenEvolveKnowledgeExportBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-import' as BubbleName,
      OpenEvolveKnowledgeImportBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-get' as BubbleName,
      OpenEvolveKnowledgeGetBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-delete' as BubbleName,
      OpenEvolveKnowledgeDeleteBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-documents' as BubbleName,
      OpenEvolveKnowledgeDocumentsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-embed' as BubbleName,
      OpenEvolveKnowledgeEmbedBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-sync' as BubbleName,
      OpenEvolveKnowledgeSyncBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-monitoring-dashboard' as BubbleName,
      OpenEvolveMonitoringDashboardBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-monitoring-alerts' as BubbleName,
      OpenEvolveMonitoringAlertsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-monitoring-services' as BubbleName,
      OpenEvolveMonitoringServicesBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-monitoring-logs' as BubbleName,
      OpenEvolveMonitoringLogsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-monitoring-metrics' as BubbleName,
      OpenEvolveMonitoringMetricsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-monitoring-health' as BubbleName,
      OpenEvolveMonitoringHealthBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-analytics-statistics' as BubbleName,
      OpenEvolveAnalyticsStatisticsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-analytics-performance' as BubbleName,
      OpenEvolveAnalyticsPerformanceBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-analytics-knowledge' as BubbleName,
      OpenEvolveAnalyticsKnowledgeBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-analytics-workflow' as BubbleName,
      OpenEvolveAnalyticsWorkflowBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-settings-icr' as BubbleName,
      OpenEvolveSettingsIcrBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-settings-determinism' as BubbleName,
      OpenEvolveSettingsDeterminismBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-settings-adaptive-decomposition' as BubbleName,
      OpenEvolveSettingsAdaptiveDecompositionBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-settings-mdap-maker' as BubbleName,
      OpenEvolveSettingsMdapMakerBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-settings-roma-mdap-maker' as BubbleName,
      OpenEvolveSettingsRomaMdapMakerBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-decomposition-to-evolution' as BubbleName,
      OpenEvolveDecompositionToEvolutionBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gauntlet-red-blue-gold' as BubbleName,
      OpenEvolveGauntletRedBlueGoldBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-adversarial-to-evolution' as BubbleName,
      OpenEvolveAdversarialToEvolutionBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-full-evolution-lifecycle' as BubbleName,
      OpenEvolveFullEvolutionLifecycleBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-workflow-decomposition-pipeline' as BubbleName,
      OpenEvolveWorkflowDecompositionPipelineBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-verified-deployment' as BubbleName,
      OpenEvolveVerifiedDeploymentBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-engine' as BubbleName,
      OpenEvolveKnowledgeEngineBubble as BubbleClassWithMetadata
    );
    this.register(
      'workflow-orchestrator' as BubbleName,
      OpenEvolveWorkflowOrchestratorBubble as BubbleClassWithMetadata
    );
    this.register(
      'ace-tools' as BubbleName,
      OpenEvolveAceToolsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-crewai' as BubbleName,
      OpenEvolveCrewAIBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-leanaide' as BubbleName,
      OpenEvolveLeanAideBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-z3prover' as BubbleName,
      OpenEvolveZ3ProverBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gauntlet-testing' as BubbleName,
      OpenEvolveGauntletTestingBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-oneke' as BubbleName,
      OpenEvolveOneKEBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gket' as BubbleName,
      OpenEvolveGKETBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-evolution-trigger' as BubbleName,
      OpenEvolveEvolutionTriggerBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-evolution-application' as BubbleName,
      OpenEvolveEvolutionApplicationBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-evolution-validation' as BubbleName,
      OpenEvolveEvolutionValidationBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-metrics-collector' as BubbleName,
      OpenEvolveMetricsCollectorBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-retrieval' as BubbleName,
      OpenEvolveKnowledgeRetrievalBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-knowledge-capture' as BubbleName,
      OpenEvolveKnowledgeCaptureBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-evolution-pipeline' as BubbleName,
      OpenEvolveEvolutionPipelineBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-continuous-evolution' as BubbleName,
      OpenEvolveContinuousEvolutionBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-adaptive-evolution' as BubbleName,
      OpenEvolveAdaptiveEvolutionBubble as BubbleClassWithMetadata
    );

    // Register OpenEvolve tool bubbles
    this.register(
      'log-parser-tool' as BubbleName,
      LogParserTool as BubbleClassWithMetadata
    );
    this.register(
      'metrics-collector-tool' as BubbleName,
      MetricsCollectorTool as BubbleClassWithMetadata
    );
    this.register(
      'vector-search-tool' as BubbleName,
      VectorSearchTool as BubbleClassWithMetadata
    );
    this.register(
      'csv-processor-tool' as BubbleName,
      CSVProcessorTool as BubbleClassWithMetadata
    );
    this.register(
      'json-validator-tool' as BubbleName,
      JSONValidatorTool as BubbleClassWithMetadata
    );
    this.register(
      'data-transformer-tool' as BubbleName,
      DataTransformerTool as BubbleClassWithMetadata
    );
    this.register(
      'file-processor-tool' as BubbleName,
      FileProcessorTool as BubbleClassWithMetadata
    );
    this.register(
      'image-processor-tool' as BubbleName,
      ImageProcessorTool as BubbleClassWithMetadata
    );
    this.register(
      'xml-parser-tool' as BubbleName,
      XMLParserTool as BubbleClassWithMetadata
    );
    this.register(
      'pdf-generator-tool' as BubbleName,
      PDFGeneratorTool as BubbleClassWithMetadata
    );
    this.register(
      'email-validator-tool' as BubbleName,
      EmailValidatorTool as BubbleClassWithMetadata
    );
    this.register(
      'url-validator-tool' as BubbleName,
      URLValidatorTool as BubbleClassWithMetadata
    );
    this.register(
      'code-formatter-tool' as BubbleName,
      CodeFormatterTool as BubbleClassWithMetadata
    );
    this.register(
      'text-analyzer-tool' as BubbleName,
      TextAnalyzerTool as BubbleClassWithMetadata
    );

    // Register OpenEvolve workflow bubbles
    this.register(
      'data-enrichment-workflow' as BubbleName,
      DataEnrichmentWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'backup-restore-workflow' as BubbleName,
      BackupRestoreWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'monitoring-alert-workflow' as BubbleName,
      MonitoringAlertWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'etl-pipeline-workflow' as BubbleName,
      ETLPipelineWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'api-aggregator-workflow' as BubbleName,
      APIAggregatorWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'scheduled-task-workflow' as BubbleName,
      ScheduledTaskWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'event-handler-workflow' as BubbleName,
      EventHandlerWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'multi-step-approval-workflow' as BubbleName,
      MultiStepApprovalWorkflow as BubbleClassWithMetadata
    );
    this.register(
      'webhook-repeater-workflow' as BubbleName,
      WebhookRepeaterWorkflow as BubbleClassWithMetadata
    );

    this.register(
      'bubblelabs-status' as BubbleName,
      OpenEvolveBubblelabsStatusBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-init' as BubbleName,
      OpenEvolveBubblelabsInitBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-workflow-definitions' as BubbleName,
      OpenEvolveBubblelabsWorkflowDefinitionsBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-workflow-instances' as BubbleName,
      OpenEvolveBubblelabsWorkflowInstancesBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-ace-skillbook' as BubbleName,
      OpenEvolveBubblelabsAceSkillbookBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-ace-patterns' as BubbleName,
      OpenEvolveBubblelabsAcePatternsBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-z3' as BubbleName,
      OpenEvolveBubblelabsZ3Bubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-roma' as BubbleName,
      OpenEvolveBubblelabsRomaBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-knowledge-store' as BubbleName,
      OpenEvolveBubblelabsKnowledgeStoreBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-knowledge-query' as BubbleName,
      OpenEvolveBubblelabsKnowledgeQueryBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-analytics' as BubbleName,
      OpenEvolveBubblelabsAnalyticsBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-leanaide-prove' as BubbleName,
      OpenEvolveBubblelabsLeanaideProveBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-integrations' as BubbleName,
      OpenEvolveBubblelabsIntegrationsBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-control-catalog' as BubbleName,
      OpenEvolveBubblelabsControlCatalogBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-control-discover' as BubbleName,
      OpenEvolveBubblelabsControlDiscoverBubble as BubbleClassWithMetadata
    );
    this.register(
      'bubblelabs-control-execute' as BubbleName,
      OpenEvolveBubblelabsControlExecuteBubble as BubbleClassWithMetadata
    );
    this.register(
      'icr-overview' as BubbleName,
      OpenEvolveIcrOverviewBubble as BubbleClassWithMetadata
    );
    this.register(
      'icr-components' as BubbleName,
      OpenEvolveIcrComponentsBubble as BubbleClassWithMetadata
    );
    this.register(
      'icr-refinements' as BubbleName,
      OpenEvolveIcrRefinementsBubble as BubbleClassWithMetadata
    );
    this.register(
      'icr-vlm' as BubbleName,
      OpenEvolveIcrVlmBubble as BubbleClassWithMetadata
    );
    this.register(
      'icr-analytics' as BubbleName,
      OpenEvolveIcrAnalyticsBubble as BubbleClassWithMetadata
    );
    this.register(
      'icr-config' as BubbleName,
      OpenEvolveIcrConfigBubble as BubbleClassWithMetadata
    );
    this.register(
      'icr-dashboard' as BubbleName,
      OpenEvolveIcrDashboardBubble as BubbleClassWithMetadata
    );
    this.register(
      'adaptive-mdap-dashboard' as BubbleName,
      OpenEvolveAdaptiveMdapDashboardBubble as BubbleClassWithMetadata
    );
    this.register(
      'adaptive-mdap-profiles' as BubbleName,
      OpenEvolveAdaptiveMdapProfilesBubble as BubbleClassWithMetadata
    );
    this.register(
      'adaptive-mdap-cost' as BubbleName,
      OpenEvolveAdaptiveMdapCostBubble as BubbleClassWithMetadata
    );
    this.register(
      'adaptive-mdap-complexity' as BubbleName,
      OpenEvolveAdaptiveMdapComplexityBubble as BubbleClassWithMetadata
    );
    this.register(
      'adaptive-mdap-allocate' as BubbleName,
      OpenEvolveAdaptiveMdapAllocateBubble as BubbleClassWithMetadata
    );
    this.register(
      'ragbits-search' as BubbleName,
      OpenEvolveRagbitsSearchBubble as BubbleClassWithMetadata
    );
    this.register(
      'ragbits-ingest' as BubbleName,
      OpenEvolveRagbitsIngestBubble as BubbleClassWithMetadata
    );
    this.register(
      'ragbits-stats' as BubbleName,
      OpenEvolveRagbitsStatsBubble as BubbleClassWithMetadata
    );
    this.register(
      'dspy-assess' as BubbleName,
      OpenEvolveDspyAssessBubble as BubbleClassWithMetadata
    );
    this.register(
      'dspy-fix' as BubbleName,
      OpenEvolveDspyFixBubble as BubbleClassWithMetadata
    );
    this.register(
      'pygraphistry' as BubbleName,
      OpenEvolvePygraphistryBubble as BubbleClassWithMetadata
    );
    this.register(
      'web3-status' as BubbleName,
      OpenEvolveWeb3StatusBubble as BubbleClassWithMetadata
    );
    this.register(
      'web3-ingest' as BubbleName,
      OpenEvolveWeb3IngestBubble as BubbleClassWithMetadata
    );
    this.register(
      'web3-slither' as BubbleName,
      OpenEvolveWeb3SlitherBubble as BubbleClassWithMetadata
    );
    this.register(
      'web3-foundry' as BubbleName,
      OpenEvolveWeb3FoundryBubble as BubbleClassWithMetadata
    );
    this.register(
      'web3-invariants' as BubbleName,
      OpenEvolveWeb3InvariantsBubble as BubbleClassWithMetadata
    );
    this.register(
      'web3-exploit-witness' as BubbleName,
      OpenEvolveWeb3ExploitWitnessBubble as BubbleClassWithMetadata
    );
    this.register(
      'web3-audit-exploit' as BubbleName,
      OpenEvolveWeb3AuditExploitBubble as BubbleClassWithMetadata
    );
    this.register(
      'web3-mcp-inventory' as BubbleName,
      OpenEvolveWeb3McpInventoryBubble as BubbleClassWithMetadata
    );
    this.register(
      'sovereign-status' as BubbleName,
      OpenEvolveSovereignStatusBubble as BubbleClassWithMetadata
    );
    this.register(
      'sovereign-problems' as BubbleName,
      OpenEvolveSovereignProblemsBubble as BubbleClassWithMetadata
    );
    this.register(
      'sovereign-plans' as BubbleName,
      OpenEvolveSovereignPlansBubble as BubbleClassWithMetadata
    );
    this.register(
      'sovereign-stats' as BubbleName,
      OpenEvolveSovereignStatsBubble as BubbleClassWithMetadata
    );
    this.register(
      'sovereign-run' as BubbleName,
      OpenEvolveSovereignRunBubble as BubbleClassWithMetadata
    );
    this.register(
      'sovereign-runs' as BubbleName,
      OpenEvolveSovereignRunsBubble as BubbleClassWithMetadata
    );
    this.register(
      'sovereign-pipeline' as BubbleName,
      OpenEvolveSovereignPipelineBubble as BubbleClassWithMetadata
    );
    this.register(
      'maker-status' as BubbleName,
      OpenEvolveMakerStatusBubble as BubbleClassWithMetadata
    );
    this.register(
      'maker-tools' as BubbleName,
      OpenEvolveMakerToolsBubble as BubbleClassWithMetadata
    );
    this.register(
      'maker-test' as BubbleName,
      OpenEvolveMakerTestBubble as BubbleClassWithMetadata
    );
    this.register(
      'maker-validate' as BubbleName,
      OpenEvolveMakerValidateBubble as BubbleClassWithMetadata
    );
    this.register(
      'maker-execute' as BubbleName,
      OpenEvolveMakerExecuteBubble as BubbleClassWithMetadata
    );
    this.register(
      'maker-delegations' as BubbleName,
      OpenEvolveMakerDelegationsBubble as BubbleClassWithMetadata
    );
    this.register(
      'maker-delegations-sync' as BubbleName,
      OpenEvolveMakerDelegationsSyncBubble as BubbleClassWithMetadata
    );
    this.register(
      'knowledge-explorer-status' as BubbleName,
      OpenEvolveKnowledgeExplorerStatusBubble as BubbleClassWithMetadata
    );
    this.register(
      'knowledge-explorer-query' as BubbleName,
      OpenEvolveKnowledgeExplorerQueryBubble as BubbleClassWithMetadata
    );
    this.register(
      'knowledge-explorer-history' as BubbleName,
      OpenEvolveKnowledgeExplorerHistoryBubble as BubbleClassWithMetadata
    );
    this.register(
      'knowledge-explorer-extract' as BubbleName,
      OpenEvolveKnowledgeExplorerExtractBubble as BubbleClassWithMetadata
    );
    this.register(
      'knowledge-explorer-extract-file' as BubbleName,
      OpenEvolveKnowledgeExplorerExtractFileBubble as BubbleClassWithMetadata
    );
    this.register(
      'suggestions-content' as BubbleName,
      OpenEvolveSuggestionsContentBubble as BubbleClassWithMetadata
    );
    this.register(
      'suggestions-classification' as BubbleName,
      OpenEvolveSuggestionsClassificationBubble as BubbleClassWithMetadata
    );
    this.register(
      'suggestions-security' as BubbleName,
      OpenEvolveSuggestionsSecurityBubble as BubbleClassWithMetadata
    );
    this.register(
      'suggestions-improvement' as BubbleName,
      OpenEvolveSuggestionsImprovementBubble as BubbleClassWithMetadata
    );

    // After all default bubbles are registered, auto-populate bubbleDependencies
    if (!BubbleFactory.dependenciesPopulated) {
      console.log('Populating bubble dependencies from source....');
      await this.populateBubbleDependenciesFromSource();
      BubbleFactory.dependenciesPopulated = true;
      // Cache detailed dependencies globally for seeding future instances
      BubbleFactory.detailedDepsCache = new Map(this.detailedDeps);
    } else {
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
  getAll(): BubbleClassWithMetadata[] {
    return Array.from(this.registry.values());
  }

  /**
   * Get metadata for a bubble without instantiating it
   */
  getMetadata(name: BubbleName) {
    const BubbleClass = this.get(name);
    if (!BubbleClass) return undefined;

    // Type guard to check if schema is a ZodObject
    const schemaParams =
      BubbleClass.schema &&
      typeof BubbleClass.schema === 'object' &&
      'shape' in BubbleClass.schema
        ? (BubbleClass.schema as z.ZodObject<z.ZodRawShape>).shape
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
  private async populateBubbleDependenciesFromSource(): Promise<void> {
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
        const ownerBubbleNames = this.extractBubbleNamesFromContent(
          content
        ) as BubbleName[];
        if (ownerBubbleNames.length === 0) {
          continue;
        }

        // Parse instances used within this file
        let instancesByDep: Map<
          BubbleName,
          {
            variableName: string;
            isAnonymous: boolean;
            startLine?: number;
            endLine?: number;
          }[]
        > = new Map();
        try {
          instancesByDep = parseBubbleInstancesFromSource(content, lookup, {
            debug: false,
            filePath,
          });
        } catch {
          // ignore parser failures for this file
        }

        // Collect ai-agent tools from instances directly (AST-derived)
        const aiAgentInst = instancesByDep.get(
          'ai-agent' as BubbleName
        ) as unknown as
          | Array<{
              variableName: string;
              isAnonymous: boolean;
              startLine?: number;
              endLine?: number;
              tools?: BubbleName[];
            }>
          | undefined;
        const aiTools = Array.from(
          new Set(
            (aiAgentInst || [])
              .flatMap((i) => i.tools || [])
              .filter((t): t is BubbleName => typeof t === 'string')
          )
        );

        for (const owner of ownerBubbleNames) {
          const detailed: BubbleDependencySpec[] = [];
          for (const [depName, instList] of instancesByDep.entries()) {
            if (depName === owner) continue;
            const spec: BubbleDependencySpec = {
              name: depName,
              instances: instList.map((i) => ({
                variableName: i.variableName,
                isAnonymous: i.isAnonymous,
                startLine: i.startLine,
                endLine: i.endLine,
              })),
            };
            if (depName === ('ai-agent' as BubbleName) && aiTools.length > 0) {
              spec.tools = aiTools as BubbleName[];
            }
            detailed.push(spec);
          }

          // Persist results for this owner bubble
          this.detailedDeps.set(owner, detailed);
          // Maintain classic flat dependency list on the class
          const klass = this.get(owner);
          if (klass) {
            try {
              (klass as any).bubbleDependencies = detailed.map((d) => d.name);
            } catch {
              try {
                Object.defineProperty(klass as object, 'bubbleDependencies', {
                  value: detailed.map((d) => d.name),
                  configurable: true,
                });
              } catch {
                // ignore
              }
            }
          }
        }
      }
    } catch {
      // Silently ignore issues in dependency scanning to avoid blocking runtime
    }
  }

  private async listModuleFilesRecursively(dir: string): Promise<string[]> {
    const out: string[] = [];
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const nested = await this.listModuleFilesRecursively(full);
        out.push(...nested);
      } else if (
        entry.isFile() &&
        (full.endsWith('.ts') || full.endsWith('.js')) &&
        !full.endsWith('.test.ts') &&
        !full.endsWith('.d.ts')
      ) {
        out.push(full);
      }
    }

    return out;
  }

  private extractBubbleNamesFromContent(content: string): string[] {
    const names: string[] = [];
    // Look for static bubbleName definitions in the class body
    const nameRegex =
      /static\s+(?:readonly\s+)?bubbleName\s*(?::[^=]+)?=\s*['"]([^'"\n]+)['"]/g;
    let match: RegExpExecArray | null;
    while ((match = nameRegex.exec(content)) !== null) {
      names.push(match[1] as BubbleName);
    }
    return names;
  }

  /**
   * Get credential to bubble name mapping from registered bubbles
   * Provides type-safe mapping based on actual registered bubbles
   */
  getCredentialToBubbleMapping(): Partial<Record<CredentialType, BubbleName>> {
    const mapping: Partial<Record<CredentialType, BubbleName>> = {};

    for (const [bubbleName, credentialOptions] of Object.entries(
      BUBBLE_CREDENTIAL_OPTIONS
    )) {
      // Get the bubble class to check its type
      const BubbleClass = this.get(bubbleName as BubbleName);

      // Only include service bubbles for credential validation
      if (BubbleClass && BubbleClass.type === 'service') {
        for (const credentialType of credentialOptions) {
          // Only map if we haven't seen this credential type before
          // This gives priority to the first service bubble for each credential
          if (!mapping[credentialType]) {
            mapping[credentialType] = bubbleName as BubbleName;
          }
        }
      }
    }

    return mapping;
  }

  /**
   * Get bubble name for a specific credential type
   */
  getBubbleNameForCredential(
    credentialType: CredentialType
  ): BubbleName | undefined {
    const mapping = this.getCredentialToBubbleMapping();
    return mapping[credentialType];
  }

  /**
   * Check if a credential type is supported by any registered bubble
   */
  isCredentialSupported(credentialType: CredentialType): boolean {
    return this.getBubbleNameForCredential(credentialType) !== undefined;
  }

  /**
   * Generate minimal BubbleFlow boilerplate template
   * Use get-trigger-detail-tool to get specific trigger configuration and payload types
   */
  generateBubbleFlowBoilerplate(options?: { className?: string }): string {
    const className = options?.className || 'GeneratedFlow';

    // Generate dynamic trigger list from registry
    const triggerList = Object.keys(TRIGGER_EVENT_CONFIGS)
      .map((t) => `'${t}'`)
      .join(' | ');

    // Dynamically generate bubble imports from registry
    const nameToClass = this.getBubbleNameToClassNameMap();
    const serviceBubbles: string[] = [];
    const toolBubbles: string[] = [];

    for (const [bubbleName, className_] of Object.entries(nameToClass)) {
      const meta = this.getMetadata(bubbleName as BubbleName);
      if (!meta) continue;
      const line = `  ${className_}, // bubble name: '${bubbleName}'`;
      if (meta.type === 'tool') {
        toolBubbles.push(line);
      } else {
        // service, workflow, and any other types go in the service section
        serviceBubbles.push(line);
      }
    }

    return `
import { z } from 'zod';
import {
  // Base classes
  BubbleFlow,

  // Service Bubbles (Connects to external services)
${serviceBubbles.join('\n')}

  // Tool Bubbles (Perform useful actions)
${toolBubbles.join('\n')}

  // RAGBits Bubbles (Semantic search and retrieval)
  RAGBitsIngestBubble, // bubble name: 'ragbits-ingest'
  RAGBitsSearchBubble, // bubble name: 'ragbits-search'
  RAGBitsIndexBubble, // bubble name: 'ragbits-index'
  RAGBitsGenerationBubble, // bubble name: 'ragbits-generation'

  // CrewAI Bubbles (Orchestration and multi-agent workflows)
  CrewAIOrchestrationBubble, // bubble name: 'crewai-orchestration'
  CrewAIResearchBubble, // bubble name: 'crewai-research'

  // Event Types (How the workflow is triggered)
  type WebhookEvent,
  type CronEvent,
  type SlackMentionEvent,
  type SlackMessageReceivedEvent,
} from '@bubblelab/bubble-core';

// AVAILABLE TRIGGERS: ${triggerList}
// Use get-trigger-detail-tool to get the payload schema and setup instructions for your chosen trigger

export interface Output {
  message: string;
  // Add your output fields here
}

export class ${className} extends BubbleFlow<'webhook/http'> {
  async handle(payload: WebhookEvent): Promise<Output> {
    // Example: instantiate a bubble and call .action() to execute it
    // const calendar = new GoogleCalendarBubble({
    //   operation: 'list_events',
    //   calendar_id: 'primary',
    //   time_min: '2025-01-01T00:00:00Z',
    //   time_max: '2025-12-31T23:59:59Z',
    // });
    // const result = await calendar.action();
    // if (!result.success) throw new Error(result.error);
    // const events = result.data?.events || [];

    // Use get-bubble to learn about each bubble's parameters and operations
    return { message: 'Hello from BubbleFlow!' };
  }
}
`;
  }
}
