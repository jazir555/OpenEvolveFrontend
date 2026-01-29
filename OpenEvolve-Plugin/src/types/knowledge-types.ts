/**
 * Knowledge Engine TypeScript Interfaces
 */

/**
 * Knowledge Engine Configuration
 */
export interface KnowledgeConfig {
  enabled: boolean;
  defaultMethod: 'pattern' | 'llm' | 'hybrid';
  confidenceThreshold: number;
  autoUpdateKnowledgeBase: boolean;
  enableKnowledgeGraphs: boolean;
  searchDepth: number;
  maxResults: number;
  includeReasoning: boolean;
  extractionSchema: string[];
  projectPath: string;
  targetStructure: string;
}

/**
 * Knowledge entity
 */
export interface KnowledgeEntity {
  id: string;
  type: string;
  label: string;
  properties: Record<string, any>;
  confidence: number;
  source?: string;
}

/**
 * Knowledge relationship
 */
export interface KnowledgeRelationship {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, any>;
  confidence: number;
}

/**
 * Knowledge extraction result
 */
export interface KnowledgeExtractionResult {
  entities: KnowledgeEntity[];
  relationships: KnowledgeRelationship[];
  confidence: number;
  metadata: {
    extractedAt: string;
    executionTime: number;
    sourcesProcessed: number;
  };
}

/**
 * Default Knowledge Configuration
 */
export const DEFAULT_KNOWLEDGE_CONFIG: KnowledgeConfig = {
  enabled: true,
  defaultMethod: 'hybrid',
  confidenceThreshold: 0.6,
  autoUpdateKnowledgeBase: true,
  enableKnowledgeGraphs: true,
  searchDepth: 2,
  maxResults: 10,
  includeReasoning: false,
  extractionSchema: [],
  projectPath: '.',
  targetStructure: 'Analyze code for concepts and relationships.'
};
