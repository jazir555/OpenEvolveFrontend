import {
  pgTable,
  text,
  serial,
  integer,
  boolean,
  timestamp,
  unique,
  jsonb,
  doublePrecision,
  index,
} from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';
import type { CredentialMetadata } from '@bubblelab/shared-schemas';

export const users = pgTable('users', {
  clerkId: text('clerk_id').primaryKey(),
  firstName: text('first_name'),
  lastName: text('last_name'),
  email: text('email').notNull(),
  appType: text('app_type').notNull().default('nodex'), // Track which app the user belongs to
  monthlyUsageCount: integer('monthly_usage_count').notNull().default(0),
  createdAt: timestamp('created_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: timestamp('updated_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const bubbleFlows = pgTable('bubble_flows', {
  id: serial().primaryKey(),
  userId: text('user_id')
    .notNull()
    .references(() => users.clerkId, { onDelete: 'cascade' }),
  name: text().notNull(),
  description: text(),
  prompt: text(), // Store the original prompt used to generate the flow (nullable)
  code: text(), // This will store the processed/transpiled code (nullable for empty flows during generation)
  originalCode: text('original_code'), // Store the original TypeScript code
  generationError: text('generation_error'), // Store any code generation errors
  bubbleParameters: jsonb('bubble_parameters'), // Store parsed bubble parameters as JSONB
  metadata: jsonb('metadata'), // Store workflow metadata (outputDescription, etc.) as JSONB
  workflow: jsonb('workflow'), // Store parsed workflow structure as JSONB
  eventType: text('event_type').notNull(),
  inputSchema: jsonb('input_schema'), // Store input schema
  webhookExecutionCount: integer('webhook_execution_count')
    .notNull()
    .default(0), // Track webhook executions
  webhookFailureCount: integer('webhook_failure_count').notNull().default(0), // Track webhook failures
  cron: text('cron'), // Cron expression extracted from code
  cronActive: boolean('cron_active').notNull().default(false), // Whether cron scheduling is active
  defaultInputs: jsonb('default_inputs'), // User-filled input values for cron execution
  createdAt: timestamp('created_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: timestamp('updated_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const webhooks = pgTable(
  'webhooks',
  {
    id: serial().primaryKey(),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    path: text('path').notNull(),
    bubbleFlowId: integer('bubble_flow_id')
      .notNull()
      .references(() => bubbleFlows.id, { onDelete: 'cascade' }),
    isActive: boolean('is_active').notNull().default(false),
    createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
    updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
  },
  (table) => ({
    // Unique combination of userId and path
    userPathUnique: unique().on(table.userId, table.path),
  })
);

export const bubbleFlowExecutions = pgTable('bubble_flow_executions', {
  id: serial().primaryKey(),
  bubbleFlowId: integer('bubble_flow_id')
    .notNull()
    .references(() => bubbleFlows.id, { onDelete: 'cascade' }),
  payload: jsonb('payload').notNull(), // JSON stored as JSONB
  result: jsonb('result'), // JSON stored as JSONB
  status: text('status').notNull(),
  error: text('error'),
  code: text('code'), // Store the original code at execution time
  executionLogs: jsonb('execution_logs'), // StreamingLogEvent[] from execution
  startedAt: timestamp('started_at', { mode: 'date' }).notNull().defaultNow(),
  completedAt: timestamp('completed_at', { mode: 'date' }),
});

export const bubbleFlowEvaluations = pgTable('bubble_flow_evaluations', {
  id: serial().primaryKey(),
  executionId: integer('execution_id')
    .notNull()
    .references(() => bubbleFlowExecutions.id, { onDelete: 'cascade' }),
  bubbleFlowId: integer('bubble_flow_id')
    .notNull()
    .references(() => bubbleFlows.id, { onDelete: 'cascade' }),
  // Evaluation result from Rice agent
  working: boolean('working').notNull(), // Whether the workflow is functioning correctly
  issueType: text('issue_type'), // 'setup' | 'workflow' | 'input' | null
  summary: text('summary').notNull(), // Brief summary of execution or issue description
  rating: integer('rating').notNull(), // Quality rating 1-10
  // Metadata
  modelUsed: text('model_used').notNull(), // Model used for evaluation (e.g., RECOMMENDED_MODELS.FAST)
  evaluatedAt: timestamp('evaluated_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const userCredentials = pgTable('user_credentials', {
  id: serial().primaryKey(),
  userId: text('user_id')
    .notNull()
    .references(() => users.clerkId, { onDelete: 'cascade' }),
  credentialType: text('credential_type').notNull(), // e.g., 'OPENAI_CRED', 'SLACK_CRED'
  encryptedValue: text('encrypted_value'), // Encrypted credential value (nullable for OAuth)
  name: text('name'), // Optional user-friendly name for the credential
  metadata: jsonb('metadata').$type<CredentialMetadata>(), // Typed JSONB field for credential metadata (DatabaseMetadata or JiraOAuthMetadata)

  // OAuth-specific fields
  oauthAccessToken: text('oauth_access_token'), // Encrypted OAuth access token
  oauthRefreshToken: text('oauth_refresh_token'), // Encrypted OAuth refresh token
  oauthExpiresAt: timestamp('oauth_expires_at', { mode: 'date' }), // Token expiration
  oauthScopes: jsonb('oauth_scopes').$type<string[]>(), // OAuth scopes granted
  oauthTokenType: text('oauth_token_type').default('Bearer'), // Token type (usually Bearer)
  oauthProvider: text('oauth_provider'), // Provider name (google, slack, github, etc.)
  isOauth: boolean('is_oauth').default(false), // Flag to identify OAuth vs API key credentials
  isDefault: boolean('is_default').default(false), // Whether this credential is the default for its type

  // BrowserBase session credential fields
  isBrowserSession: boolean('is_browser_session').default(false), // Flag for browser session credentials
  browserbaseContextId: text('browserbase_context_id'), // Context ID for session persistence
  browserbaseCookies: text('browserbase_cookies'), // Encrypted JSON cookies array
  browserbaseSessionData: jsonb('browserbase_session_data').$type<{
    capturedAt: string;
    cookieCount: number;
    domain: string;
  }>(), // Session metadata

  createdAt: timestamp('created_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: timestamp('updated_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const userServiceUsage = pgTable(
  'user_service_usage',
  {
    id: serial().primaryKey(),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    service: text('service').notNull(), // CredentialType enum value (e.g., 'OPENAI_CRED', 'FIRECRAWL_API_KEY')
    subService: text('sub_service'), // Optional: e.g., 'gpt-4', 'gemini-2.0-flash', 'apify/instagram-scraper'
    monthYear: text('month_year').notNull(), // e.g., '2025-01'
    unit: text('unit').notNull(), // e.g., 'per_1m_tokens', 'per_email', 'per_result'
    usage: doublePrecision('usage').notNull().default(0), // Usage count in the specified unit (high precision float)
    unitCost: doublePrecision('unit_cost').notNull(), // Cost per unit in dollars (high precision float)
    totalCost: doublePrecision('total_cost').notNull().default(0), // Calculated: usage * unitCost (high precision float)
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: timestamp('updated_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    // Unique constraint: one record per user, service, subService, and unit
    userServiceUnitUnique: unique().on(
      table.userId,
      table.service,
      table.subService,
      table.unit,
      table.unitCost,
      table.monthYear
    ),
  })
);

export const waitlistedUsers = pgTable('waitlisted_users', {
  email: text('email').primaryKey(),
  name: text('name').notNull(),
  database: text('database').notNull(), // e.g., 'postgres', 'mysql', 'mongodb', etc.
  otherDatabase: text('other_database'), // For when database is 'other'
  status: text('status').notNull().default('pending'), // 'pending', 'approved', 'rejected', 'converted'
  notes: text('notes'), // Admin notes about the user
  createdAt: timestamp('created_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: timestamp('updated_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const bubbleFlowsRelations = relations(bubbleFlows, ({ many }) => ({
  executions: many(bubbleFlowExecutions),
  webhooks: many(webhooks),
  evaluations: many(bubbleFlowEvaluations),
}));

export const webhooksRelations = relations(webhooks, ({ one }) => ({
  bubbleFlow: one(bubbleFlows, {
    fields: [webhooks.bubbleFlowId],
    references: [bubbleFlows.id],
  }),
}));

export const bubbleFlowExecutionsRelations = relations(
  bubbleFlowExecutions,
  ({ one, many }) => ({
    bubbleFlow: one(bubbleFlows, {
      fields: [bubbleFlowExecutions.bubbleFlowId],
      references: [bubbleFlows.id],
    }),
    evaluations: many(bubbleFlowEvaluations),
  })
);

export const bubbleFlowEvaluationsRelations = relations(
  bubbleFlowEvaluations,
  ({ one }) => ({
    execution: one(bubbleFlowExecutions, {
      fields: [bubbleFlowEvaluations.executionId],
      references: [bubbleFlowExecutions.id],
    }),
    bubbleFlow: one(bubbleFlows, {
      fields: [bubbleFlowEvaluations.bubbleFlowId],
      references: [bubbleFlows.id],
    }),
  })
);

// No relations needed for userCredentials as it's a standalone table

// ============================================================================
// EVOLUTION TABLES
// ============================================================================

export const evolutionRequests = pgTable(
  'evolution_requests',
  {
    id: serial().primaryKey(),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    content: text('content').notNull(),
    mode: text('mode').notNull().default('standard'),
    parameters: jsonb('parameters'),
    constraints: jsonb('constraints'),
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: timestamp('updated_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    userIdIdx: index('evolution_requests_user_id_idx').on(table.userId),
  })
);

export const evolutionDesigns = pgTable(
  'evolution_designs',
  {
    id: serial().primaryKey(),
    requestId: integer('request_id')
      .notNull()
      .references(() => evolutionRequests.id, { onDelete: 'cascade' }),
    generation: integer('generation').notNull(),
    html: text('html').notNull(),
    css: text('css'),
    metadata: jsonb('metadata'),
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    requestGenerationIdx: index(
      'evolution_designs_request_generation_idx'
    ).on(table.requestId, table.generation),
  })
);

export const evolutionJudgeScores = pgTable(
  'evolution_judge_scores',
  {
    id: serial().primaryKey(),
    designId: integer('design_id')
      .notNull()
      .references(() => evolutionDesigns.id, { onDelete: 'cascade' }),
    agent: text('agent').notNull(),
    score: doublePrecision('score').notNull(),
    reasoning: text('reasoning'),
    highlights: jsonb('highlights'),
    issues: jsonb('issues'),
    recommendations: jsonb('recommendations'),
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    designIdIdx: index('evolution_judge_scores_design_id_idx').on(
      table.designId
    ),
  })
);

export const evolutionResults = pgTable(
  'evolution_results',
  {
    id: serial().primaryKey(),
    requestId: integer('request_id')
      .notNull()
      .references(() => evolutionRequests.id, { onDelete: 'cascade' }),
    bestDesignId: integer('best_design_id').references(
      () => evolutionDesigns.id,
      { onDelete: 'set null' }
    ),
    summary: jsonb('summary'),
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    requestIdIdx: index('evolution_results_request_id_idx').on(
      table.requestId
    ),
  })
);

export const evolutionRuns = pgTable(
  'evolution_runs',
  {
    id: serial().primaryKey(),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    evolutionId: text('evolution_id').notNull(),
    status: text('status').notNull(),
    name: text('name'),
    config: jsonb('config'),
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: timestamp('updated_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    userEvolutionUnique: unique().on(table.userId, table.evolutionId),
  })
);

export const evolutionAssets = pgTable('evolution_assets', {
  id: serial().primaryKey(),
  runId: integer('run_id')
    .notNull()
    .references(() => evolutionRuns.id, { onDelete: 'cascade' }),
  userId: text('user_id')
    .notNull()
    .references(() => users.clerkId, { onDelete: 'cascade' }),
  kind: text('kind').notNull(),
  contentType: text('content_type').notNull(),
  filePath: text('file_path').notNull(),
  size: integer('size').notNull(),
  createdAt: timestamp('created_at', { mode: 'date' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const evolutionNodes = pgTable(
  'evolution_nodes',
  {
    id: serial().primaryKey(),
    runId: integer('run_id')
      .notNull()
      .references(() => evolutionRuns.id, { onDelete: 'cascade' }),
    nodeId: text('node_id').notNull(),
    parentNodeId: text('parent_node_id'),
    generation: integer('generation').notNull(),
    status: text('status').notNull(),
    fitness: doublePrecision('fitness'),
    score: doublePrecision('score'),
    label: text('label'),
    htmlAssetId: integer('html_asset_id').references(() => evolutionAssets.id, {
      onDelete: 'set null',
    }),
    thumbnailAssetId: integer('thumbnail_asset_id').references(
      () => evolutionAssets.id,
      { onDelete: 'set null' }
    ),
    metadata: jsonb('metadata'),
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: timestamp('updated_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    runNodeUnique: unique().on(table.runId, table.nodeId),
  })
);

export const evolutionScreenshots = pgTable(
  'evolution_screenshots',
  {
    id: serial().primaryKey(),
    designId: integer('design_id')
      .notNull()
      .references(() => evolutionDesigns.id, { onDelete: 'cascade' }),
    assetId: integer('asset_id').references(() => evolutionAssets.id, {
      onDelete: 'set null',
    }),
    kind: text('kind').notNull().default('thumbnail'),
    width: integer('width'),
    height: integer('height'),
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    designIdIdx: index('evolution_screenshots_design_id_idx').on(
      table.designId
    ),
  })
);

export const idempotencyKeys = pgTable(
  'idempotency_keys',
  {
    id: serial().primaryKey(),
    key: text('key').notNull(),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    endpoint: text('endpoint'),
    params: jsonb('params'),
    response: jsonb('response'),
    statusCode: integer('status_code').notNull(),
    expiresAt: timestamp('expires_at', { mode: 'date' }),
    createdAt: timestamp('created_at', { mode: 'date' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    keyUserUnique: unique().on(table.key, table.userId),
  })
);

export const evolutionRequestsRelations = relations(
  evolutionRequests,
  ({ many }) => ({
    designs: many(evolutionDesigns),
    results: many(evolutionResults),
  })
);

export const evolutionDesignsRelations = relations(
  evolutionDesigns,
  ({ one, many }) => ({
    request: one(evolutionRequests, {
      fields: [evolutionDesigns.requestId],
      references: [evolutionRequests.id],
    }),
    judgeScores: many(evolutionJudgeScores),
    screenshots: many(evolutionScreenshots),
  })
);

export const evolutionJudgeScoresRelations = relations(
  evolutionJudgeScores,
  ({ one }) => ({
    design: one(evolutionDesigns, {
      fields: [evolutionJudgeScores.designId],
      references: [evolutionDesigns.id],
    }),
  })
);

export const evolutionResultsRelations = relations(
  evolutionResults,
  ({ one }) => ({
    request: one(evolutionRequests, {
      fields: [evolutionResults.requestId],
      references: [evolutionRequests.id],
    }),
    bestDesign: one(evolutionDesigns, {
      fields: [evolutionResults.bestDesignId],
      references: [evolutionDesigns.id],
    }),
  })
);

export const evolutionRunsRelations = relations(
  evolutionRuns,
  ({ many }) => ({
    nodes: many(evolutionNodes),
    assets: many(evolutionAssets),
  })
);

export const evolutionAssetsRelations = relations(
  evolutionAssets,
  ({ one, many }) => ({
    run: one(evolutionRuns, {
      fields: [evolutionAssets.runId],
      references: [evolutionRuns.id],
    }),
    screenshots: many(evolutionScreenshots),
    htmlNodes: many(evolutionNodes, { relationName: 'assetHtml' }),
    thumbnailNodes: many(evolutionNodes, {
      relationName: 'assetThumbnail',
    }),
  })
);

export const evolutionNodesRelations = relations(
  evolutionNodes,
  ({ one }) => ({
    run: one(evolutionRuns, {
      fields: [evolutionNodes.runId],
      references: [evolutionRuns.id],
    }),
    htmlAsset: one(evolutionAssets, {
      fields: [evolutionNodes.htmlAssetId],
      references: [evolutionAssets.id],
      relationName: 'assetHtml',
    }),
    thumbnailAsset: one(evolutionAssets, {
      fields: [evolutionNodes.thumbnailAssetId],
      references: [evolutionAssets.id],
      relationName: 'assetThumbnail',
    }),
  })
);

export const evolutionScreenshotsRelations = relations(
  evolutionScreenshots,
  ({ one }) => ({
    design: one(evolutionDesigns, {
      fields: [evolutionScreenshots.designId],
      references: [evolutionDesigns.id],
    }),
    asset: one(evolutionAssets, {
      fields: [evolutionScreenshots.assetId],
      references: [evolutionAssets.id],
    }),
  })
);

// No relations needed for idempotencyKeys as it's a standalone table
