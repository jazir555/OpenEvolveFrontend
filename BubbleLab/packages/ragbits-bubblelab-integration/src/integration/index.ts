// Export processor integration functionality

export * from './RagbitsProcessorIntegration';

// Export main processor class
import { RAGBitsDocumentProcessor } from './RagbitsProcessorIntegration';
export { RAGBitsDocumentProcessor };

// Export integration utilities
export function createProcessorIntegration(config?: any): RAGBitsDocumentProcessor {
  return new RAGBitsDocumentProcessor(config);
}