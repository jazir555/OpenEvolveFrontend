/**
 * Ragbits + BubbleLab Integration - Main Entry Point
 * 
 * This module exports all components for the Ragbits + BubbleLab integration
 */

// Export Bubble components
export { BaseBubble } from './bubbles/BaseBubble';
export { RAGBitsIngestBubble, type RAGBitsIngestConfig, type RAGBitsIngestInput, type RAGBitsIngestOutput } from './bubbles/RAGBitsIngestBubble';
export { RAGBitsSearchBubble, type RAGBitsSearchConfig, type RAGBitsSearchInput, type RAGBitsSearchOutput } from './bubbles/RAGBitsSearchBubble';
export { RAGBitsGenerationBubble, type RAGBitsGenerationConfig, type RAGBitsGenerationInput, type RAGBitsGenerationOutput } from './bubbles/RAGBitsGenerationBubble';
export { RAGBitsIndexBubble, type RAGBitsIndexConfig, type RAGBitsIndexInput, type RAGBitsIndexOutput } from './bubbles/RAGBitsIndexBubble';

// Export configuration mapper
export { 
  ConfigMapper, 
  type BubbleLabWorkflowConfig, 
  type BubbleLabNode, 
  type BubbleLabEdge, 
  type RagbitsConfig, 
  type RagbitsNodeConfig, 
  type RagbitsConnection 
} from './config/config_mapper';

// Export configuration generator
export { ConfigGenerator, type GenerationOptions, type GeneratedConfig } from './config/config_generator';

// Export workflow engine
export { RAGBitsWorkflowEngine, type WorkflowExecutionResult, type WorkflowExecutionOptions } from './engine/ragbits_workflow_engine';

// Export processor integration
export { 
  RagbitsProcessorIntegration, 
  type ProcessorIntegrationConfig, 
  type ProcessedDocument, 
  type ProcessingStats 
} from './integration/ragbits_processor_integration';

// Export monitoring service
export { 
  MonitoringService, 
  type MonitoringEvent, 
  type PerformanceMetrics, 
  type DebugInfo, 
  type MonitoringConfig 
} from './monitoring/monitoring_service';

// Export all types
export * from './types';

// Export main integration class
export class RagbitsBubbleLabIntegration {
  private static instance: RagbitsBubbleLabIntegration;
  
  static getInstance(): RagbitsBubbleLabIntegration {
    if (!RagbitsBubbleLabIntegration.instance) {
      RagbitsBubbleLabIntegration.instance = new RagbitsBubbleLabIntegration();
    }
    return RagbitsBubbleLabIntegration.instance;
  }

  /**
   * Creates a new RAG workflow engine
   */
  createWorkflowEngine(workflowConfig: BubbleLabWorkflowConfig, options?: WorkflowExecutionOptions) {
    return new RAGBitsWorkflowEngine(workflowConfig, options);
  }

  /**
   * Creates a new processor integration
   */
  createProcessorIntegration(config?: ProcessorIntegrationConfig) {
    return new RagbitsProcessorIntegration(config);
  }

  /**
   * Creates a new monitoring service
   */
  createMonitoringService(config?: Partial<MonitoringConfig>) {
    return new MonitoringService(config);
  }

  /**
   * Generates a Ragbits configuration from a BubbleLab workflow
   */
  generateConfig(bubbleLabConfig: BubbleLabWorkflowConfig, options?: GenerationOptions) {
    return ConfigGenerator.generate(bubbleLabConfig, options);
  }

  /**
   * Maps a BubbleLab workflow to Ragbits configuration
   */
  mapConfig(bubbleLabConfig: BubbleLabWorkflowConfig) {
    return ConfigMapper.mapBubbleLabToRagbits(bubbleLabConfig);
  }
}

// Export the default instance
export default RagbitsBubbleLabIntegration.getInstance();