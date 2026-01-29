// Export all type definitions

export * from './bubble-config';
export * from './input-output';
export * from './monitoring-debug';

// Additional utility types

export type { RAGBitsIngestConfig } from './bubble-config';
export type { RAGBitsSearchConfig } from './bubble-config';
export type { RAGBitsGenerationConfig } from './bubble-config';
export type { RAGBitsIndexConfig } from './bubble-config';
export type { BubbleConfig } from './bubble-config';
export type { BubbleLabNode } from './bubble-config';
export type { BubbleLabEdge } from './bubble-config';
export type { BubbleLabWorkflowConfig } from './bubble-config';
export type { RagbitsNodeConfig } from './bubble-config';
export type { RagbitsConnection } from './bubble-config';
export type { RagbitsConfig } from './bubble-config';

export type { RAGBitsIngestInput } from './input-output';
export type { RAGBitsIngestOutput } from './input-output';
export type { RAGBitsSearchInput } from './input-output';
export type { RAGBitsSearchOutput } from './input-output';
export type { RAGBitsGenerationInput } from './input-output';
export type { RAGBitsGenerationOutput } from './input-output';
export type { RAGBitsIndexInput } from './input-output';
export type { RAGBitsIndexOutput } from './input-output';
export type { ProcessedDocument } from './input-output';
export type { ProcessingStats } from './input-output';
export type { WorkflowExecutionResult } from './input-output';
export type { WorkflowExecutionOptions } from './input-output';
export type { GenerationOptions } from './input-output';
export type { GeneratedConfig } from './input-output';

export type { MonitoringEvent } from './monitoring-debug';
export type { PerformanceMetrics } from './monitoring-debug';
export type { DebugInfo } from './monitoring-debug';
export type { MonitoringConfig } from './monitoring-debug';
export type { ProcessorIntegrationConfig } from './monitoring-debug';

// Type validation utilities
export function isRAGBitsIngestConfig(config: any): config is RAGBitsIngestConfig {
  return config && typeof config.sourceType === 'string' && 
         (config.sourceType === 'file' || config.sourceType === 'url' || config.sourceType === 'text');
}

export function isRAGBitsSearchConfig(config: any): config is RAGBitsSearchConfig {
  return config && typeof config.searchStrategy === 'string' && 
         typeof config.topK === 'number';
}

export function isRAGBitsGenerationConfig(config: any): config is RAGBitsGenerationConfig {
  return config && typeof config.model === 'string';
}

export function isRAGBitsIndexConfig(config: any): config is RAGBitsIndexConfig {
  return config && Array.isArray(config.operations);
}

export function isBubbleLabWorkflowConfig(config: any): config is BubbleLabWorkflowConfig {
  return config && typeof config.id === 'string' && 
         typeof config.name === 'string' && 
         Array.isArray(config.nodes) && 
         Array.isArray(config.edges);
}