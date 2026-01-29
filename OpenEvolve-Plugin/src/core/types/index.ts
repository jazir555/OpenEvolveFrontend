// Plugin types
export * from './plugin';

// Workflow types
export interface Workflow {
  id: string;
  name: string;
  description?: string;
  type: string;
  config: Record<string, unknown>;
  status: 'idle' | 'running' | 'completed' | 'failed';
  createdAt: string;
  updatedAt: string;
  executedAt?: string;
  result?: unknown;
  error?: string;
}

export interface WorkflowExecution {
  workflowId: string;
  executionId: string;
  status: 'running' | 'completed' | 'failed';
  startedAt: string;
  completedAt?: string;
  result?: unknown;
  error?: string;
  logs: ExecutionLog[];
}

export interface ExecutionLog {
  timestamp: string;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  data?: unknown;
}

// Analytics types
export interface AnalyticsMetric {
  name: string;
  value: number;
  unit?: string;
  change?: number;
  changeType?: 'increase' | 'decrease';
}

export interface Artifact {
  id: string;
  type: string;
  name: string;
  description?: string;
  content: unknown;
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

// Knowledge base types
export interface KnowledgeArtifact {
  id: string;
  type: string;
  title: string;
  description?: string;
  content: unknown;
  tags: string[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  version: number;
}

export interface KnowledgeSearchResult {
  artifacts: KnowledgeArtifact[];
  total: number;
  page: number;
  pageSize: number;
}

// LeanAide types
export interface LeanProof {
  id: string;
  theorem: string;
  status: 'pending' | 'proving' | 'verified' | 'failed';
  model: string;
  proof?: string;
  error?: string;
  progress: number;
  createdAt: string;
  updatedAt: string;
}

export interface LeanVerification {
  proofId: string;
  status: 'pending' | 'verifying' | 'verified' | 'failed';
  errors: string[];
  warnings: string[];
  progress: number;
}
