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
      'openevolve-slack' as BubbleName,
      OpenEvolveSlackBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-gmail' as BubbleName,
      OpenEvolveGmailBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-http' as BubbleName,
      OpenEvolveHttpBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-github' as BubbleName,
      OpenEvolveGithubBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-apify' as BubbleName,
      OpenEvolveApifyBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-google-drive' as BubbleName,
      OpenEvolveGoogleDriveBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-google-sheets' as BubbleName,
      OpenEvolveGoogleSheetsBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-airtable' as BubbleName,
      OpenEvolveAirtableBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-notion' as BubbleName,
      OpenEvolveNotionBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-postgresql' as BubbleName,
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
      'openevolve-knowledge-engine' as BubbleName,
      OpenEvolveKnowledgeEngineBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-workflow-orchestrator' as BubbleName,
      OpenEvolveWorkflowOrchestratorBubble as BubbleClassWithMetadata
    );
    this.register(
      'openevolve-ace-tools' as BubbleName,
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
