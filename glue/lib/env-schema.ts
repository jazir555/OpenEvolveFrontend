/**
 * Centralized Environment Variable Schema
 *
 * Following the Federation Constitution:
 * - Law of Configuration Explicitness: No magic defaults
 * - All configurable values must be via environment variables
 * - Crashes immediately if required vars are missing
 *
 * This is the SINGLE SOURCE OF TRUTH for all environment variables.
 * Corresponds to: glue/ENVIRONMENT_VARIABLES.md and .env.schema
 */

import { EnvVar } from './env-validator';

/**
 * Core OpenEvolve Environment Variables
 */
export const CORE_ENV_VARS: EnvVar[] = [
  // Logging & Telemetry
  { name: 'OPENEVOLVE_LOG_LEVEL', type: 'string', required: false, default: 'INFO' },
  { name: 'LOG_LEVEL', type: 'string', required: false, default: 'INFO' },
  { name: 'LOG_FORMAT', type: 'string', required: false, default: 'json' },

  // Service Ports
  { name: 'OPENEVOLVE_ORCHESTRATOR_PORT', type: 'port', required: false, default: 8080 },
  { name: 'ORCHESTRATOR_PORT', type: 'port', required: false, default: 8080 },

  // Service Toggles
  { name: 'OPENEVOLVE_SERVICES__REST_API', type: 'boolean', required: false, default: true },
  { name: 'OPENEVOLVE_SERVICES__GRAPHQL_API', type: 'boolean', required: false, default: true },
  { name: 'OPENEVOLVE_SERVICES__EVENT_BUS', type: 'boolean', required: false, default: true },
  { name: 'OPENEVOLVE_SERVICES__MCP_SERVER', type: 'boolean', required: false, default: true },
  { name: 'OPENEVOLVE_SERVICES__TELEMETRY', type: 'boolean', required: false, default: true },

  // REST API Configuration
  { name: 'OPENEVOLVE_REST_API__HOST', type: 'string', required: false, default: '0.0.0.0' },
  { name: 'OPENEVOLVE_REST_API__PORT', type: 'port', required: false, default: 8000 },
  { name: 'OPENEVOLVE_REST_API__CORS_ORIGINS', type: 'string', required: false, default: '*' },
  { name: 'MAX_REQUEST_SIZE', type: 'number', required: false, default: 100 },
  { name: 'TIMEOUT', type: 'number', required: false, default: 300 },

  // GraphQL Configuration
  { name: 'OPENEVOLVE_GRAPHQL__HOST', type: 'string', required: false, default: '0.0.0.0' },
  { name: 'OPENEVOLVE_GRAPHQL__PORT', type: 'port', required: false, default: 8001 },
  { name: 'OPENEVOLVE_GRAPHQL__ENABLE_PLAYGROUND', type: 'boolean', required: false, default: true },

  // Rate Limiting
  { name: 'RATE_LIMIT_REQUESTS_PER_MINUTE', type: 'number', required: false, default: 100 },
  { name: 'RATE_LIMIT_BURST_SIZE', type: 'number', required: false, default: 10 },
  { name: 'RATE_LIMIT_ENABLED', type: 'boolean', required: false, default: true },
  { name: 'RATE_LIMIT_PER_MINUTE', type: 'number', required: false, default: 100 },
  { name: 'RATE_LIMIT_BURST', type: 'number', required: false, default: 10 },

  // Security
  { name: 'SECRET_KEY', type: 'string', required: true },
  { name: 'JWT_ALGORITHM', type: 'string', required: false, default: 'HS256' },
  { name: 'ACCESS_TOKEN_EXPIRE_MINUTES', type: 'number', required: false, default: 30 },
  { name: 'REFRESH_TOKEN_EXPIRE_DAYS', type: 'number', required: false, default: 7 },
  { name: 'API_KEY', type: 'string', required: false },

  // Worker Configuration
  { name: 'WORKERS', type: 'number', required: false, default: 0 },

  // Development Settings
  { name: 'DEBUG', type: 'boolean', required: false, default: false },
  { name: 'RELOAD', type: 'boolean', required: false, default: false },
  { name: 'NODE_ENV', type: 'string', required: false, default: 'production' },
];

/**
 * Infrastructure Environment Variables
 */
export const INFRA_ENV_VARS: EnvVar[] = [
  // Valkey/Redis Configuration
  { name: 'VALKEY_HOST', type: 'string', required: false, default: 'localhost' },
  { name: 'VALKEY_PORT', type: 'port', required: false, default: 6379 },
  { name: 'VALKEY_PASSWORD', type: 'string', required: false },
  { name: 'REDIS_HOST', type: 'string', required: false, default: 'localhost' },
  { name: 'REDIS_PORT', type: 'port', required: false, default: 6379 },
  { name: 'REDIS_DB', type: 'number', required: false, default: 0 },
  { name: 'REDIS_PASSWORD', type: 'string', required: false },

  // Database Configuration
  { name: 'DATABASE_URL', type: 'url', required: false },
  { name: 'DB_HOST', type: 'string', required: false, default: 'localhost' },
  { name: 'DB_PORT', type: 'port', required: false, default: 5432 },
  { name: 'DB_USERNAME', type: 'string', required: false, default: 'openevolve' },
  { name: 'DB_PASSWORD', type: 'string', required: true },
  { name: 'DB_NAME', type: 'string', required: false, default: 'openevolve_kg' },

  // Backup Configuration
  { name: 'BACKUP_DIR', type: 'string', required: false, default: './backups' },
  { name: 'BACKUP_RETENTION_DAYS', type: 'number', required: false, default: 30 },
];

/**
 * API Gateway Environment Variables
 */
export const API_GATEWAY_ENV_VARS: EnvVar[] = [
  { name: 'API_HOST', type: 'string', required: false, default: '0.0.0.0' },
  { name: 'API_PORT', type: 'port', required: false, default: 8000 },
  { name: 'API_RELOAD', type: 'boolean', required: false, default: false },

  // Clerk JWT Configuration
  { name: 'CLERK_ISSUER', type: 'url', required: false },
  { name: 'CLERK_JWKS_URL', type: 'url', required: false },
  { name: 'CLERK_AUDIENCE', type: 'string', required: false },
  { name: 'CLERK_JWKS_CACHE_TTL_SECONDS', type: 'number', required: false, default: 3600 },

  // CORS Configuration
  { name: 'CORS_ORIGINS', type: 'string', required: false, default: '["http://localhost:3000","http://localhost:8000"]' },
  { name: 'CORS_ALLOW_CREDENTIALS', type: 'boolean', required: false, default: true },
  { name: 'CORS_ALLOW_METHODS', type: 'string', required: false, default: '["*"]' },
  { name: 'CORS_ALLOW_HEADERS', type: 'string', required: false, default: '["*"]' },

  // File Upload Configuration
  { name: 'MAX_FILE_SIZE', type: 'number', required: false, default: 10485760 },
  { name: 'UPLOAD_DIR', type: 'string', required: false, default: './uploads' },

  // WebSocket Configuration
  { name: 'WS_HEARTBEAT_INTERVAL', type: 'number', required: false, default: 30 },
  { name: 'WS_MAX_CONNECTIONS', type: 'number', required: false, default: 100 },

  // Evolution Orchestrator
  { name: 'EVOLUTION_ORCHESTRATOR_URL', type: 'url', required: false, default: 'http://localhost:8003/evolve' },
];

/**
 * Adapter Environment Variables
 */
export const BUBBLELAB_ADAPTER_ENV_VARS: EnvVar[] = [
  { name: 'BUBBLELAB_PORT', type: 'port', required: false, default: 3001 },
  { name: 'BUBBLELAB_API_URL', type: 'url', required: true },
  { name: 'BUBBLELAB_API_KEY', type: 'string', required: true },
  { name: 'BUBBLELAB_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
  { name: 'BUBBLELAB_MAX_RETRIES', type: 'number', required: false, default: 3 },
];

export const GRAPHITI_ADAPTER_ENV_VARS: EnvVar[] = [
  { name: 'GRAPHITI_PORT', type: 'port', required: false, default: 3000 },
  { name: 'NEO4J_URI', type: 'url', required: false, default: 'bolt://localhost:7687' },
  { name: 'NEO4J_USER', type: 'string', required: false, default: 'neo4j' },
  { name: 'NEO4J_PASSWORD', type: 'string', required: true },
  { name: 'GRAPHITI_API_URL', type: 'url', required: false, default: 'http://localhost:8000' },
  { name: 'GRAPHITI_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
  { name: 'OPENAI_API_KEY', type: 'string', required: false },
  { name: 'ANTHROPIC_API_KEY', type: 'string', required: false },
  { name: 'UPDATE_COMMUNITIES', type: 'boolean', required: false, default: false },
  { name: 'STORE_RAW_EPISODES', type: 'boolean', required: false, default: true },
];

export const VECTORDB_ADAPTER_ENV_VARS: EnvVar[] = [
  { name: 'VECTORDB_PORT', type: 'port', required: false, default: 3004 },
  { name: 'VECTORDB_TYPE', type: 'string', required: false, default: 'pinecone' },
  { name: 'VECTORDB_API_URL', type: 'url', required: true },
  { name: 'VECTORDB_API_KEY', type: 'string', required: true },
  { name: 'VECTORDB_CONNECTION_STRING', type: 'string', required: false },
  { name: 'VECTORDB_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
  { name: 'VECTORDB_MAX_RETRIES', type: 'number', required: false, default: 3 },
  { name: 'PINECONE_ENVIRONMENT', type: 'string', required: false },

  // General defaults
  { name: 'TIMEOUT_MS', type: 'number', required: false, default: 5000 },
  { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
];

export const OPENEVOLVE_ADAPTER_ENV_VARS: EnvVar[] = [
  { name: 'OPENEVOLVE_PORT', type: 'port', required: false, default: 3003 },
  { name: 'OPENEVOLVE_API_URL', type: 'url', required: true },
  { name: 'OPENEVOLVE_API_KEY', type: 'string', required: true },
  { name: 'OPENEVOLVE_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
  { name: 'OPENEVOLVE_MAX_RETRIES', type: 'number', required: false, default: 3 },
  { name: 'DEFAULT_REQUEST_TIMEOUT', type: 'number', required: false, default: 30000 },
];

export const ICR_ADAPTER_ENV_VARS: EnvVar[] = [
  { name: 'ICR_PORT', type: 'port', required: false, default: 3002 },
  { name: 'ICR_API_URL', type: 'url', required: true },
  { name: 'ICR_API_KEY', type: 'string', required: true },
  { name: 'ICR_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
  { name: 'ICR_MAX_RETRIES', type: 'number', required: false, default: 3 },
  { name: 'ICR_RETRY_DELAY_MS', type: 'number', required: false, default: 1000 },
];

export const LEANAIDE_ADAPTER_ENV_VARS: EnvVar[] = [
  { name: 'LEANAIDE_PORT', type: 'port', required: false, default: 3006 },
  { name: 'LEANAIDE_API_URL', type: 'url', required: true },
  { name: 'LEANAIDE_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
  { name: 'LEANAIDE_MAX_RETRIES', type: 'number', required: false, default: 3 },
];

export const Z3_ADAPTER_ENV_VARS: EnvVar[] = [
  { name: 'Z3_PORT', type: 'port', required: false, default: 3005 },
  { name: 'Z3_SOLVER_PATH', type: 'string', required: false, default: '/usr/bin/z3' },
  { name: 'Z3_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
  { name: 'Z3_MAX_MEMORY_MB', type: 'number', required: false, default: 4096 },
  { name: 'Z3_MAX_RETRIES', type: 'number', required: false, default: 3 },
  { name: 'Z3_API_URL', type: 'url', required: false },
  { name: 'Z3_HEALTH_CHECK', type: 'string', required: false, default: '/health' },
  { name: 'Z3_VERIFY_PATH', type: 'string', required: false, default: '/verify' },
];

/**
 * RESE Adapter Environment Variables
 */
export const RESE_DEE_ENV_VARS: EnvVar[] = [
  { name: 'RESE_DEE_PORT', type: 'port', required: false, default: 8001 },
  { name: 'RESE_DEE_EXPLORATION_DEPTH', type: 'number', required: false, default: 10 },
  { name: 'RESE_DEE_MCTS_ITERATIONS', type: 'number', required: false, default: 1000 },
  { name: 'RESE_DEE_MCTS_EXPLORATION_CONSTANT', type: 'number', required: false, default: 1.414 },
  { name: 'RESE_DEE_CONVERGENCE_THRESHOLD', type: 'number', required: false, default: 0.001 },
  { name: 'RESE_DEE_EXPLORATION_TIMEOUT_MS', type: 'number', required: false, default: 10000 },
  { name: 'RESE_DEE_MAX_HYPOTHESES', type: 'number', required: false, default: 100 },
  { name: 'RESE_DEE_PATTERN_RECOGNITION_THRESHOLD', type: 'number', required: false, default: 0.7 },
];

export const RESE_LLTDL_ENV_VARS: EnvVar[] = [
  { name: 'RESE_LLTDL_PORT', type: 'port', required: false, default: 8002 },
  { name: 'RESE_LLTDL_ENCODING_DIM', type: 'number', required: false, default: 128 },
  { name: 'RESE_LLTDL_USE_POSITIONAL', type: 'boolean', required: false, default: true },
  { name: 'RESE_LLTDL_USE_TYPE_EMBEDDING', type: 'boolean', required: false, default: true },
  { name: 'RESE_LLTDL_USE_CATEGORY_EMBEDDING', type: 'boolean', required: false, default: true },
  { name: 'RESE_LLTDL_MAX_SEQUENCE_LENGTH', type: 'number', required: false, default: 512 },
  { name: 'RESE_LLTDL_CACHE_SIZE', type: 'number', required: false, default: 1000 },
  { name: 'RESE_LLTDL_DEFAULT_LOSS_TYPE', type: 'string', required: false, default: 'mse' },
  { name: 'RESE_LLTDL_COMBINATION_STRATEGY', type: 'string', required: false, default: 'weighted_sum' },
  { name: 'RESE_LLTDL_NORMALIZE_WEIGHTS', type: 'boolean', required: false, default: true },
  { name: 'RESE_LLTDL_LEARNING_RATE', type: 'number', required: false, default: 0.001 },
  { name: 'RESE_LLTDL_TIMEOUT_MS', type: 'number', required: false, default: 3000 },
  { name: 'RESE_LLTDL_ENABLE_RTREE', type: 'boolean', required: false, default: false },
  { name: 'RESE_LLTDL_ENABLE_LSH', type: 'boolean', required: false, default: false },
  { name: 'RESE_LLTDL_ENABLE_HAG', type: 'boolean', required: false, default: false },
  { name: 'RESE_LLTDL_CONTRADICTION_THRESHOLD', type: 'number', required: false, default: 0.8 },
  { name: 'RESE_LLTDL_MAX_CONTRADICTIONS', type: 'number', required: false, default: 1000 },
];

export const RESE_SCE_ENV_VARS: EnvVar[] = [
  { name: 'RESE_SCE_PORT', type: 'port', required: false, default: 8003 },
  { name: 'SCE_TIMEOUT_MS', type: 'number', required: false, default: 5000 },
  { name: 'SCE_CONSTRAINT_TIMEOUT_MS', type: 'number', required: false, default: 3000 },
  { name: 'SCE_CONTRADICTION_DETECTION_TIMEOUT_MS', type: 'number', required: false, default: 10000 },
  { name: 'SCE_MAX_ITERATIONS', type: 'number', required: false, default: 1000 },
  { name: 'SCE_MAX_CONSTRAINTS', type: 'number', required: false, default: 10000 },
  { name: 'SCE_MAX_CONTRADICTION_SET_SIZE', type: 'number', required: false, default: 100 },
  { name: 'SCE_CIRCUIT_BREAKER_THRESHOLD', type: 'number', required: false, default: 5 },
  { name: 'SCE_CIRCUIT_BREAKER_TIMEOUT_MS', type: 'number', required: false, default: 60000 },
  { name: 'SCE_ENABLE_LEAN4_INTEGRATION', type: 'boolean', required: false, default: false },
  { name: 'SCE_ENABLE_TACIT_MINING', type: 'boolean', required: false, default: true },
  { name: 'SCE_CONTRADICTION_DETECTION', type: 'boolean', required: false, default: true },
  { name: 'SCE_FORMAL_VERIFICATION', type: 'boolean', required: false, default: true },
];

export const RESE_PHASE_ENV_VARS: EnvVar[] = [
  { name: 'RESE_PHASE2_PORT', type: 'port', required: false, default: 8004 },
  { name: 'RESE_PHASE4_PORT', type: 'port', required: false, default: 8006 },
  { name: 'RESE_VERIFICATION_PORT', type: 'port', required: false, default: 8007 },
  { name: 'RESE_Z3_BRIDGE_PORT', type: 'port', required: false, default: 8008 },
  { name: 'RESE_LEANAIDE_WORKFLOW_PORT', type: 'port', required: false, default: 8009 },
  { name: 'RESE_LEANAIDE_API_URL', type: 'url', required: false, default: 'http://leanaide-core:8000' },
  { name: 'RESE_LEANAIDE_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
  { name: 'RESE_LEANAIDE_MAX_RETRIES', type: 'number', required: false, default: 3 },
];

/**
 * Knowledge Engine Environment Variables
 */
export const KNOWLEDGE_ENGINE_ENV_VARS: EnvVar[] = [
  { name: 'VECTOR_STORE_HOST', type: 'string', required: false, default: 'localhost' },
  { name: 'VECTOR_STORE_PORT', type: 'port', required: false, default: 6333 },
  { name: 'CACHE_HOST', type: 'string', required: false, default: 'localhost' },
  { name: 'CACHE_PORT', type: 'port', required: false, default: 6379 },
  { name: 'CACHE_DB', type: 'number', required: false, default: 0 },
  { name: 'SERVER_HOST', type: 'string', required: false, default: '0.0.0.0' },
  { name: 'SERVER_PORT', type: 'port', required: false, default: 8000 },
  { name: 'SERVER_WORKERS', type: 'number', required: false, default: 4 },
  { name: 'LLM_PROVIDER', type: 'string', required: false, default: 'openai' },
  { name: 'LLM_MODEL', type: 'string', required: false, default: 'gpt-4o' },
  { name: 'LLM_API_KEY', type: 'string', required: true },
  { name: 'LLM_BASE_URL', type: 'url', required: false },
  { name: 'JWT_SECRET', type: 'string', required: true },
  { name: 'KE_VALIDATE_CONFIG', type: 'string', required: false, default: 'warn' },
];

/**
 * Plugin Environment Variables
 */
export const PLUGIN_ENV_VARS: EnvVar[] = [
  { name: 'VITE_OPENEVOLVE_API_URL', type: 'url', required: false, default: 'http://localhost:8000' },
];

export const DATAPIZZA_PLUGIN_ENV_VARS: EnvVar[] = [
  { name: 'DATAPIZZA_API_URL', type: 'url', required: false, default: '/api/datapizza' },
  { name: 'DATAPIZZA_API_KEY', type: 'string', required: false },
  { name: 'DATAPIZZA_TIMEOUT', type: 'number', required: false, default: 30000 },
];

/**
 * LLM Environment Variables (used by multiple components)
 */
export const LLM_ENV_VARS: EnvVar[] = [
  { name: 'OPENAI_API_KEY', type: 'string', required: false },
  { name: 'ANTHROPIC_API_KEY', type: 'string', required: false },
  { name: 'GOOGLE_API_KEY', type: 'string', required: false },
  { name: 'GEMINI_API_KEY', type: 'string', required: false },
  { name: 'AI_API_KEY', type: 'string', required: false },
  { name: 'API_KEY', type: 'string', required: false },
  { name: 'OPENROUTER_API_KEY', type: 'string', required: false },
];

/**
 * Event Bus Environment Variables
 */
export const EVENT_BUS_ENV_VARS: EnvVar[] = [
  { name: 'EVENT_BUS_TYPE', type: 'string', required: false, default: 'memory' },
  { name: 'EVENT_BUS_URL', type: 'url', required: false },
  { name: 'OPENEVOLVE_EVENT_BUS__ENABLED', type: 'boolean', required: false, default: true },
  { name: 'OPENEVOLVE_EVENT_BUS__BACKEND', type: 'string', required: false, default: 'valkey' },
  { name: 'OPENEVOLVE_EVENT_BUS__HOST', type: 'string', required: false, default: 'localhost' },
  { name: 'OPENEVOLVE_EVENT_BUS__PORT', type: 'port', required: false, default: 6379 },
  { name: 'OPENEVOLVE_EVENT_BUS__PASSWORD', type: 'string', required: false },
  { name: 'EVENT_BUS_MAX_EVENTS', type: 'number', required: false, default: 10000 },
  { name: 'EVENT_BUS_PERSIST_EVENTS', type: 'boolean', required: false, default: true },

  // Unified Knowledge Query
  { name: 'RAGBITS_URL', type: 'url', required: false },
  { name: 'GRAPHITI_URL', type: 'url', required: false },
  { name: 'VECTORDB_URL', type: 'url', required: false },
];

/**
 * Observability Environment Variables
 */
export const OBSERVABILITY_ENV_VARS: EnvVar[] = [
  { name: 'OPENEVOLVE_TELEMETRY__ENABLED', type: 'boolean', required: false, default: true },
  { name: 'OPENEVOLVE_TELEMETRY__SERVICE_NAME', type: 'string', required: false, default: 'openevolve' },
  { name: 'OPENEVOLVE_TELEMETRY__OTLP_ENDPOINT', type: 'url', required: false, default: 'http://localhost:4317' },
  { name: 'OPENEVOLVE_TELEMETRY__METRICS_ENABLED', type: 'boolean', required: false, default: true },
  { name: 'OPENEVOLVE_TELEMETRY__TRACING_ENABLED', type: 'boolean', required: false, default: true },
  { name: 'PROMETHEUS_PORT', type: 'port', required: false, default: 9090 },
  { name: 'OTEL_EXPORTER_OTLP_ENDPOINT', type: 'url', required: false, default: 'http://localhost:4317' },
  { name: 'SERVICE_NAME', type: 'string', required: false, default: 'unknown-service' },
  { name: 'METRICS_PREFIX', type: 'string', required: false, default: 'openevolve_' },
  { name: 'ENABLE_TRACING', type: 'boolean', required: false, default: false },
];

/**
 * PES (Prompt Evolution Strategy) Environment Variables
 */
export const PES_ENV_VARS: EnvVar[] = [
  { name: 'PES_COST_OPTIMIZATION', type: 'boolean', required: false, default: false },
  { name: 'PES_MAX_COST_USD', type: 'number', required: false, default: 10.0 },
  { name: 'PES_COST_WARNING', type: 'number', required: false, default: 0.7 },
  { name: 'PES_COST_CRITICAL', type: 'number', required: false, default: 0.9 },
  { name: 'PES_EARLY_STOPPING', type: 'boolean', required: false, default: true },
  { name: 'PES_STOPPING_PATIENCE', type: 'number', required: false, default: 5 },
  { name: 'PES_MIN_IMPROVEMENT', type: 'number', required: false, default: 0.001 },
  { name: 'PES_PLANNING', type: 'boolean', required: false, default: true },
  { name: 'PES_SUMMARIZATION', type: 'boolean', required: false, default: true },
  { name: 'PES_AUTO_SELECT', type: 'boolean', required: false, default: true },
  { name: 'PES_USE_CHEAP_MODELS', type: 'boolean', required: false, default: true },
  { name: 'PES_CHEAP_MODEL', type: 'string', required: false, default: 'gpt-3.5-turbo' },
  { name: 'PES_EXPENSIVE_MODEL', type: 'string', required: false, default: 'gpt-4o' },
  { name: 'PES_PROMPT_TOKEN_PRICE', type: 'number', required: false, default: 0.00001 },
  { name: 'PES_COMPLETION_TOKEN_PRICE', type: 'number', required: false, default: 0.00003 },
];

/**
 * Orchestration Environment Variables
 */
export const ORCHESTRATION_ENV_VARS: EnvVar[] = [
  { name: 'PIPELINE_TIMEOUT_MS', type: 'number', required: false, default: 300000 },
  { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
  { name: 'CIRCUIT_BREAKER_THRESHOLD', type: 'number', required: false, default: 5 },
  { name: 'CIRCUIT_BREAKER_TIMEOUT_MS', type: 'number', required: false, default: 60000 },
  { name: 'CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS', type: 'number', required: false, default: 3 },
  { name: 'DLQ_MAX_SIZE', type: 'number', required: false, default: 1000 },
];

/**
 * All Environment Variables Combined
 * Useful for validating the entire configuration at once
 */
export const ALL_ENV_VARS: EnvVar[] = [
  ...CORE_ENV_VARS,
  ...INFRA_ENV_VARS,
  ...API_GATEWAY_ENV_VARS,
  ...BUBBLELAB_ADAPTER_ENV_VARS,
  ...GRAPHITI_ADAPTER_ENV_VARS,
  ...VECTORDB_ADAPTER_ENV_VARS,
  ...OPENEVOLVE_ADAPTER_ENV_VARS,
  ...ICR_ADAPTER_ENV_VARS,
  ...LEANAIDE_ADAPTER_ENV_VARS,
  ...Z3_ADAPTER_ENV_VARS,
  ...RESE_DEE_ENV_VARS,
  ...RESE_LLTDL_ENV_VARS,
  ...RESE_SCE_ENV_VARS,
  ...RESE_PHASE_ENV_VARS,
  ...KNOWLEDGE_ENGINE_ENV_VARS,
  ...PLUGIN_ENV_VARS,
  ...DATAPIZZA_PLUGIN_ENV_VARS,
  ...LLM_ENV_VARS,
  ...EVENT_BUS_ENV_VARS,
  ...OBSERVABILITY_ENV_VARS,
  ...PES_ENV_VARS,
  ...ORCHESTRATION_ENV_VARS,
];

/**
 * Helper function to get schema by component name
 */
export function getSchemaForComponent(componentName: string): EnvVar[] {
  const schemas: Record<string, EnvVar[]> = {
    core: CORE_ENV_VARS,
    infra: INFRA_ENV_VARS,
    apiGateway: API_GATEWAY_ENV_VARS,
    bubblelab: BUBBLELAB_ADAPTER_ENV_VARS,
    graphiti: GRAPHITI_ADAPTER_ENV_VARS,
    vectordb: VECTORDB_ADAPTER_ENV_VARS,
    openevolve: OPENEVOLVE_ADAPTER_ENV_VARS,
    icr: ICR_ADAPTER_ENV_VARS,
    leanaide: LEANAIDE_ADAPTER_ENV_VARS,
    z3: Z3_ADAPTER_ENV_VARS,
    reseDee: RESE_DEE_ENV_VARS,
    reseLltl: RESE_LLTDL_ENV_VARS,
    reseSce: RESE_SCE_ENV_VARS,
    resePhase: RESE_PHASE_ENV_VARS,
    knowledgeEngine: KNOWLEDGE_ENGINE_ENV_VARS,
    plugin: PLUGIN_ENV_VARS,
    datapizza: DATAPIZZA_PLUGIN_ENV_VARS,
    llm: LLM_ENV_VARS,
    eventBus: EVENT_BUS_ENV_VARS,
    observability: OBSERVABILITY_ENV_VARS,
    pes: PES_ENV_VARS,
    orchestration: ORCHESTRATION_ENV_VARS,
  };

  return schemas[componentName] || [];
}
