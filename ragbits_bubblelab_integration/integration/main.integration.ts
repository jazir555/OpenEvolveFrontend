/**
 * Main Integration Class for Ragbits + BubbleLab Integration
 *
 * This class provides the primary interface for the integration
 */

import { RAGBitsWorkflowEngine } from '../engine/ragbits_workflow_engine';
import { RagbitsProcessorIntegration } from '../integration/ragbits_processor_integration';
import { MonitoringService } from '../monitoring/monitoring_service';
import { ConfigGenerator } from '../config/config_generator';
import { ConfigMapper } from '../config/config_mapper';
import { 
  type BubbleLabWorkflowConfig, 
  type WorkflowExecutionOptions, 
  type ProcessorIntegrationConfig, 
  type MonitoringConfig, 
  type GenerationOptions 
} from '../types';

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