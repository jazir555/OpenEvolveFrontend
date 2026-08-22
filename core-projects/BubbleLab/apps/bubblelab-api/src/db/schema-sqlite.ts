import { sqliteTable, text, int, unique, real, index } from 'drizzle-orm/sqlite-core';
import { relations } from 'drizzle-orm';
import type { CredentialMetadata } from '@bubblelab/shared-schemas';

export const users = sqliteTable('users', {
  clerkId: text('clerk_id').primaryKey(),
  firstName: text('first_name'),
  lastName: text('last_name'),
  email: text('email').notNull(),
  appType: text('app_type').notNull().default('nodex'), // Track which app the user belongs to
  monthlyUsageCount: int('monthly_usage_count').notNull().default(0),
  createdAt: int('created_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: int('updated_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const bubbleFlows = sqliteTable('bubble_flows', {
  id: int().primaryKey({ autoIncrement: true }),
  userId: text('user_id')
    .notNull()
    .references(() => users.clerkId, { onDelete: 'cascade' }),
  name: text().notNull(),
  description: text(),
  prompt: text(), // Store the original prompt used to generate the flow (nullable)
  code: text(), // This will store the processed/transpiled code (nullable for empty flows during generation)
  originalCode: text('original_code'), // Store the original TypeScript code
  generationError: text('generation_error'), // Store any code generation errors
  bubbleParameters: text('bubble_parameters', { mode: 'json' }), // Store parsed bubble parameters
  metadata: text('metadata', { mode: 'json' }), // Store workflow metadata (outputDescription, etc.)
  workflow: text('workflow', { mode: 'json' }), // Store parsed workflow structure
  eventType: text('event_type').notNull(),
  inputSchema: text('input_schema', { mode: 'json' }), // Store input schema
  webhookExecutionCount: int('webhook_execution_count').notNull().default(0), // Track webhook executions
  webhookFailureCount: int('webhook_failure_count').notNull().default(0), // Track webhook failures
  cron: text('cron'), // Cron expression extracted from code
  cronActive: int('cron_active', { mode: 'boolean' }).notNull().default(false), // Whether cron scheduling is active
  defaultInputs: text('default_inputs', { mode: 'json' }), // User-filled input values for cron execution
  createdAt: int('created_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: int('updated_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const webhooks = sqliteTable(
  'webhooks',
  {
    id: int().primaryKey({ autoIncrement: true }),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    path: text('path').notNull(),
    bubbleFlowId: int('bubble_flow_id')
      .notNull()
      .references(() => bubbleFlows.id, { onDelete: 'cascade' }),
    isActive: int('is_active', { mode: 'boolean' }).notNull().default(false),
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: int('updated_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    // Unique combination of userId and path
    userPathUnique: unique().on(table.userId, table.path),
  })
);

export const bubbleFlowExecutions = sqliteTable('bubble_flow_executions', {
  id: int().primaryKey({ autoIncrement: true }),
  bubbleFlowId: int('bubble_flow_id')
    .notNull()
    .references(() => bubbleFlows.id, { onDelete: 'cascade' }),
  payload: text('payload', { mode: 'json' }).notNull(),
  result: text('result', { mode: 'json' }),
  status: text('status').notNull(),
  error: text('error'),
  code: text('code'), // Store the original code at execution time
  executionLogs: text('execution_logs', { mode: 'json' }), // StreamingLogEvent[] from execution
  startedAt: int('started_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
  completedAt: int('completed_at', { mode: 'timestamp' }),
});

export const bubbleFlowEvaluations = sqliteTable('bubble_flow_evaluations', {
  id: int().primaryKey({ autoIncrement: true }),
  executionId: int('execution_id')
    .notNull()
    .references(() => bubbleFlowExecutions.id, { onDelete: 'cascade' }),
  bubbleFlowId: int('bubble_flow_id')
    .notNull()
    .references(() => bubbleFlows.id, { onDelete: 'cascade' }),
  // Evaluation result from Rice agent
  working: int('working', { mode: 'boolean' }).notNull(), // Whether the workflow is functioning correctly
  issueType: text('issue_type'), // 'setup' | 'workflow' | 'input' | null
  summary: text('summary').notNull(), // Brief summary of execution or issue description
  rating: int('rating').notNull(), // Quality rating 1-10
  // Metadata
  modelUsed: text('model_used').notNull(), // Model used for evaluation (e.g., RECOMMENDED_MODELS.FAST)
  evaluatedAt: int('evaluated_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const userCredentials = sqliteTable('user_credentials', {
  id: int().primaryKey({ autoIncrement: true }),
  userId: text('user_id')
    .notNull()
    .references(() => users.clerkId, { onDelete: 'cascade' }),
  credentialType: text('credential_type').notNull(), // e.g., 'OPENAI_CRED', 'SLACK_CRED'
  encryptedValue: text('encrypted_value'), // Encrypted credential value (nullable for OAuth)
  name: text('name'), // Optional user-friendly name for the credential
  metadata: text('metadata', { mode: 'json' }).$type<CredentialMetadata>(), // Typed JSON field for credential metadata (DatabaseMetadata or JiraOAuthMetadata)

  // OAuth-specific fields
  oauthAccessToken: text('oauth_access_token'), // Encrypted OAuth access token
  oauthRefreshToken: text('oauth_refresh_token'), // Encrypted OAuth refresh token
  oauthExpiresAt: int('oauth_expires_at', { mode: 'timestamp' }), // Token expiration
  oauthScopes: text('oauth_scopes', { mode: 'json' }).$type<string[]>(), // OAuth scopes granted
  oauthTokenType: text('oauth_token_type').default('Bearer'), // Token type (usually Bearer)
  oauthProvider: text('oauth_provider'), // Provider name (google, slack, github, etc.)
  isOauth: int('is_oauth', { mode: 'boolean' }).default(false), // Flag to identify OAuth vs API key credentials
  isDefault: int('is_default', { mode: 'boolean' }).default(false), // Whether this credential is the default for its type

  // Browser session-specific fields (for BrowserBase-based authentication)
  isBrowserSession: int('is_browser_session', { mode: 'boolean' }).default(
    false
  ), // Flag for browser session credentials
  browserbaseContextId: text('browserbase_context_id'), // BrowserBase context ID for session persistence
  browserbaseCookies: text('browserbase_cookies'), // Encrypted captured cookies JSON
  browserbaseSessionData: text('browserbase_session_data', {
    mode: 'json',
  }).$type<{
    capturedAt: string;
    cookieCount: number;
    domain: string;
  }>(),

  createdAt: int('created_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: int('updated_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const userServiceUsage = sqliteTable(
  'user_service_usage',
  {
    id: int().primaryKey({ autoIncrement: true }),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    service: text('service').notNull(), // CredentialType enum value (e.g., 'OPENAI_CRED', 'FIRECRAWL_API_KEY')
    subService: text('sub_service'), // Optional: e.g., 'gpt-4', 'gemini-2.0-flash', 'apify/instagram-scraper'
    monthYear: text('month_year').notNull(), // e.g., '2025-01'
    unit: text('unit').notNull(), // e.g., 'per_1m_tokens', 'per_email', 'per_result'
    usage: real('usage').notNull().default(0), // Usage count in the specified unit (high precision float)
    unitCost: real('unit_cost').notNull(), // Cost per unit in dollars (high precision float)
    totalCost: real('total_cost').notNull().default(0), // Calculated: usage * unitCost (high precision float)
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: int('updated_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    // Unique constraint: one record per user, service, subService, and unit
    userServiceUnitUnique: unique().on(
      table.userId,
      table.service,
      table.subService,
      table.unitCost,
      table.unit,
      table.monthYear
    ),
  })
);

export const waitlistedUsers = sqliteTable('waitlisted_users', {
  email: text('email').primaryKey(),
  name: text('name').notNull(),
  database: text('database').notNull(), // e.g., 'postgres', 'mysql', 'mongodb', etc.
  otherDatabase: text('other_database'), // For when database is 'other'
  status: text('status').notNull().default('pending'), // 'pending', 'approved', 'rejected', 'converted'
  notes: text('notes'), // Admin notes about the user
  createdAt: int('created_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: int('updated_at', { mode: 'timestamp' })
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

export const evolutionRequests = sqliteTable(
  'evolution_requests',
  {
    id: int().primaryKey({ autoIncrement: true }),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    content: text('content').notNull(),
    mode: text('mode').notNull().default('standard'),
    parameters: text('parameters'),
    constraints: text('constraints'),
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: int('updated_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    userIdIdx: index('evolution_requests_user_id_idx').on(table.userId),
  })
);

export const evolutionDesigns = sqliteTable(
  'evolution_designs',
  {
    id: int().primaryKey({ autoIncrement: true }),
    requestId: int('request_id')
      .notNull()
      .references(() => evolutionRequests.id, { onDelete: 'cascade' }),
    generation: int('generation').notNull(),
    html: text('html').notNull(),
    css: text('css'),
    metadata: text('metadata'),
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    requestGenerationIdx: index(
      'evolution_designs_request_generation_idx'
    ).on(table.requestId, table.generation),
  })
);

export const evolutionJudgeScores = sqliteTable(
  'evolution_judge_scores',
  {
    id: int().primaryKey({ autoIncrement: true }),
    designId: int('design_id')
      .notNull()
      .references(() => evolutionDesigns.id, { onDelete: 'cascade' }),
    agent: text('agent').notNull(),
    score: real('score').notNull(),
    reasoning: text('reasoning'),
    highlights: text('highlights'),
    issues: text('issues'),
    recommendations: text('recommendations'),
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    designIdIdx: index('evolution_judge_scores_design_id_idx').on(
      table.designId
    ),
  })
);

export const evolutionResults = sqliteTable(
  'evolution_results',
  {
    id: int().primaryKey({ autoIncrement: true }),
    requestId: int('request_id')
      .notNull()
      .references(() => evolutionRequests.id, { onDelete: 'cascade' }),
    bestDesignId: int('best_design_id').references(() => evolutionDesigns.id, {
      onDelete: 'set null',
    }),
    summary: text('summary'),
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    requestIdIdx: index('evolution_results_request_id_idx').on(
      table.requestId
    ),
  })
);

export const evolutionRuns = sqliteTable(
  'evolution_runs',
  {
    id: int().primaryKey({ autoIncrement: true }),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    evolutionId: text('evolution_id').notNull(),
    status: text('status').notNull(),
    name: text('name'),
    config: text('config', { mode: 'json' }),
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: int('updated_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    userEvolutionUnique: unique().on(table.userId, table.evolutionId),
  })
);

export const evolutionAssets = sqliteTable('evolution_assets', {
  id: int().primaryKey({ autoIncrement: true }),
  runId: int('run_id')
    .notNull()
    .references(() => evolutionRuns.id, { onDelete: 'cascade' }),
  userId: text('user_id')
    .notNull()
    .references(() => users.clerkId, { onDelete: 'cascade' }),
  kind: text('kind').notNull(),
  contentType: text('content_type').notNull(),
  filePath: text('file_path').notNull(),
  size: int('size').notNull(),
  createdAt: int('created_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const evolutionNodes = sqliteTable(
  'evolution_nodes',
  {
    id: int().primaryKey({ autoIncrement: true }),
    runId: int('run_id')
      .notNull()
      .references(() => evolutionRuns.id, { onDelete: 'cascade' }),
    nodeId: text('node_id').notNull(),
    parentNodeId: text('parent_node_id'),
    generation: int('generation').notNull(),
    status: text('status').notNull(),
    fitness: real('fitness'),
    score: real('score'),
    label: text('label'),
    htmlAssetId: int('html_asset_id').references(() => evolutionAssets.id, {
      onDelete: 'set null',
    }),
    thumbnailAssetId: int('thumbnail_asset_id').references(
      () => evolutionAssets.id,
      { onDelete: 'set null' }
    ),
    metadata: text('metadata', { mode: 'json' }),
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
    updatedAt: int('updated_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    runNodeUnique: unique().on(table.runId, table.nodeId),
  })
);

export const evolutionScreenshots = sqliteTable(
  'evolution_screenshots',
  {
    id: int().primaryKey({ autoIncrement: true }),
    designId: int('design_id')
      .notNull()
      .references(() => evolutionDesigns.id, { onDelete: 'cascade' }),
    assetId: int('asset_id').references(() => evolutionAssets.id, {
      onDelete: 'set null',
    }),
    kind: text('kind').notNull().default('thumbnail'),
    width: int('width'),
    height: int('height'),
    createdAt: int('created_at', { mode: 'timestamp' })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    designIdIdx: index('evolution_screenshots_design_id_idx').on(
      table.designId
    ),
  })
);

export const idempotencyKeys = sqliteTable(
  'idempotency_keys',
  {
    id: int().primaryKey({ autoIncrement: true }),
    key: text('key').notNull(),
    userId: text('user_id')
      .notNull()
      .references(() => users.clerkId, { onDelete: 'cascade' }),
    endpoint: text('endpoint'),
    params: text('params', { mode: 'json' }),
    response: text('response', { mode: 'json' }),
    statusCode: int('status_code').notNull(),
    expiresAt: int('expires_at', { mode: 'timestamp' }),
    createdAt: int('created_at', { mode: 'timestamp' })
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
