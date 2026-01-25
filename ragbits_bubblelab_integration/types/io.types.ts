/**
 * Input/Output Types for Ragbits + BubbleLab Integration
 */

export interface RAGBitsIngestInput {
  source?: string;
  content?: string;
  metadata?: Record<string, any>;
}

export interface RAGBitsIngestOutput {
  success: boolean;
  documentId: string;
  chunksIngested: number;
  processingTime: number;
  error?: string;
}

export interface RAGBitsSearchInput {
  query: string;
  topK?: number;
  scoreThreshold?: number;
  filters?: Record<string, any>;
}

export interface RAGBitsSearchOutput {
  success: boolean;
  results: Array<{
    documentId: string;
    content: string;
    score: number;
    metadata: Record<string, any>;
  }>;
  totalResults: number;
  executionTime: number;
  error?: string;
}

export interface RAGBitsGenerationInput {
  query: string;
  context?: Array<{
    content: string;
    metadata: Record<string, any>;
    score?: number;
  }>;
  customPrompt?: string;
}

export interface RAGBitsGenerationOutput {
  success: boolean;
  response: string;
  executionTime: number;
  tokensUsed?: {
    input: number;
    output: number;
    total: number;
  };
  error?: string;
}

export interface RAGBitsIndexInput {
  operation: 'stats' | 'refresh' | 'clear' | 'optimize';
  force?: boolean;
}

export interface RAGBitsIndexOutput {
  success: boolean;
  operation: string;
  result: any;
  executionTime: number;
  error?: string;
}

export interface ProcessedDocument {
  id: string;
  source: string;
  content: string;
  metadata: Record<string, any>;
  processingTime: number;
  success: boolean;
  error?: string;
}

export interface ProcessingStats {
  totalProcessed: number;
  successful: number;
  failed: number;
  averageProcessingTime: number;
  lastProcessed: Date;
  queueSize: number;
}

export interface WorkflowExecutionResult {
  success: boolean;
  nodeId: string;
  output: any;
  executionTime: number;
  error?: string;
}

export interface WorkflowExecutionOptions {
  timeout?: number; // in milliseconds
  maxRetries?: number;
  enableLogging?: boolean;
  logLevel?: 'info' | 'debug' | 'warn' | 'error';
}