/**
 * OpenEvolve Plugin Constants
 */

export const API_ENDPOINTS = {
  BASE: '/api/openevolve',
  WEBSOCKET: '/ws/openevolve',
  WORKFLOWS: '/api/openevolve/workflows',
  ANALYTICS: '/api/openevolve/analytics',
  KNOWLEDGE: '/api/openevolve/knowledge',
  LEANAIDE: '/api/openevolve/leanaide',
  EVOLUTION: '/api/openevolve/evolution',
  ADVERSARIAL: '/api/openevolve/adversarial',
  MAKER: '/api/openevolve/maker',
  MDAP: '/api/openevolve/mdap',
  DECOMPOSITION: '/api/openevolve/decomposition',
  CREWAI: '/api/openevolve/crewai',
  ROMA: '/api/openevolve/roma',
  INVENTION: '/api/openevolve/invention',
} as const;

export const WORKFLOW_TYPES = {
  EVOLUTION: 'evolution',
  ADVERSARIAL: 'adversarial',
  MAKER: 'maker',
  MDAP: 'mdap',
  DECOMPOSITION: 'decomposition',
  INVENTION: 'invention',
} as const;

export const EXECUTION_STATUS = {
  IDLE: 'idle',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
} as const;

export const ARTIFACT_TYPES = {
  MODEL: 'model',
  DATASET: 'dataset',
  PROOF: 'proof',
  WORKFLOW: 'workflow',
  RESULT: 'result',
  LOG: 'log',
} as const;

export const LEAN_MODELS = {
  MATHLIB: 'mathlib',
  STD: 'std',
  ALEAN: 'alean',
  COUNTEREXAMPLES: 'counterexamples',
} as const;

export const PROOF_STATUS = {
  PENDING: 'pending',
  PROVING: 'proving',
  VERIFIED: 'verified',
  FAILED: 'failed',
} as const;

export const THEME_COLORS = {
  PRIMARY: '#3b82f6',
  SUCCESS: '#10b981',
  WARNING: '#f59e0b',
  ERROR: '#ef4444',
  INFO: '#6366f1',
} as const;

export const CHART_COLORS = [
  '#3b82f6',
  '#10b981',
  '#f59e0b',
  '#ef4444',
  '#6366f1',
  '#8b5cf6',
  '#ec4899',
  '#14b8a6',
] as const;

export const DEFAULT_PAGINATION = {
  PAGE: 1,
  PAGE_SIZE: 20,
} as const;

export const WEBSOCKET_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  ERROR: 'error',
  WORKFLOW_STARTED: 'workflow.started',
  WORKFLOW_UPDATED: 'workflow.updated',
  WORKFLOW_COMPLETED: 'workflow.completed',
  WORKFLOW_FAILED: 'workflow.failed',
  LOG_MESSAGE: 'log.message',
  PROOF_PROGRESS: 'proof.progress',
  ANALYTICS_UPDATE: 'analytics.update',
} as const;

export const LOCAL_STORAGE_KEYS = {
  AUTH_TOKEN: 'openevolve_auth_token',
  USER_PREFERENCES: 'openevolve_preferences',
  RECENT_WORKFLOWS: 'openevolve_recent_workflows',
  SAVED_QUERIES: 'openevolve_saved_queries',
} as const;
