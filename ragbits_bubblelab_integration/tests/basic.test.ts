/**
 * Basic Tests for Ragbits + BubbleLab Integration
 */

import { 
  RagbitsBubbleLabIntegration,
  BaseBubble,
  RAGBitsIngestBubble,
  RAGBitsSearchBubble,
  RAGBitsGenerationBubble,
  RAGBitsIndexBubble,
  ConfigMapper,
  ConfigGenerator,
  RAGBitsWorkflowEngine,
  RagbitsProcessorIntegration,
  MonitoringService
} from '../index';

describe('Basic Integration Tests', () => {
  test('should import all components successfully', () => {
    expect(RagbitsBubbleLabIntegration).toBeDefined();
    expect(BaseBubble).toBeDefined();
    expect(RAGBitsIngestBubble).toBeDefined();
    expect(RAGBitsSearchBubble).toBeDefined();
    expect(RAGBitsGenerationBubble).toBeDefined();
    expect(RAGBitsIndexBubble).toBeDefined();
    expect(ConfigMapper).toBeDefined();
    expect(ConfigGenerator).toBeDefined();
    expect(RAGBitsWorkflowEngine).toBeDefined();
    expect(RagbitsProcessorIntegration).toBeDefined();
    expect(MonitoringService).toBeDefined();
  });

  test('should create integration singleton', () => {
    const instance1 = RagbitsBubbleLabIntegration.getInstance();
    const instance2 = RagbitsBubbleLabIntegration.getInstance();
    
    expect(instance1).toBe(instance2);
  });

  test('should have proper type definitions', () => {
    // This test verifies that our type definitions are properly exported
    const config = {
      id: 'test',
      name: 'Test',
      description: 'Test config',
      sourceType: 'file' as const,
      sourcePath: './test.txt'
    };
    
    const bubble = new RAGBitsIngestBubble(config);
    expect(bubble).toBeDefined();
  });
});