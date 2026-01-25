/**
 * Knowledge Extraction Node
 *
 * Extracts structured knowledge entities and relationships from input sources.
 * Provides a deterministic fallback extractor when no external model is supplied.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema,
} from './OpenEvolveBaseNode';
import { knowledgeApi } from '../services/api/endpoints';

export type KnowledgeSourceType = 'text' | 'database' | 'api' | 'file' | 'graph' | 'semantic' | 'custom';
export type ExtractionMethod = 'pattern' | 'ml' | 'llm' | 'hybrid';

export interface KnowledgeSource {
  id: string;
  type: KnowledgeSourceType;
  content: string;
  metadata?: Record<string, any>;
}

export interface KnowledgeEntity {
  id: string;
  type: string;
  label: string;
  properties: Record<string, any>;
  confidence: number;
  source: string;
}

export interface KnowledgeRelationship {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, any>;
  confidence: number;
}

export interface KnowledgeExtractionResult {
  entities: KnowledgeEntity[];
  relationships: KnowledgeRelationship[];
  confidence: number;
  metadata: {
    extractedAt: Date;
    executionTime: number;
    sourcesProcessed: number;
  };
}

export interface KnowledgeExtractionNodeConfig {
  sources?: KnowledgeSource[];
  method?: ExtractionMethod;
  domain?: string;
  validateKnowledge?: boolean;
  confidenceThreshold?: number;
  updateKnowledgeBase?: boolean;
  maxDepth?: number;
  useKnowledgeGraphs?: boolean;
  graphTraversalStrategy?: 'breadth' | 'depth' | 'best';
  extractionSchema?: string[];
}

export class KnowledgeExtractionNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Knowledge Extraction';
  static readonly DESCRIPTION = 'Extract entities and relationships from knowledge sources';
  static readonly ICON = 'knowledge';
  static readonly CATEGORY = 'knowledge';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: KnowledgeExtractionNodeConfig = {}) {
    super(id, {
      method: 'pattern',
      domain: 'general',
      validateKnowledge: true,
      confidenceThreshold: 0.6,
      updateKnowledgeBase: false,
      maxDepth: 2,
      useKnowledgeGraphs: false,
      graphTraversalStrategy: 'breadth',
      ...config,
    });
  }

  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();
      const sources = (inputs.sources as KnowledgeSource[]) ||
        (this.config.sources as KnowledgeSource[]) ||
        [];

      if (!sources.length) {
        return this.createErrorResult('At least one knowledge source is required');
      }

      context.updateProgress(20, 'Extracting entities via Knowledge Engine');
      
      let entities: KnowledgeEntity[] = [];
      let relationships: KnowledgeRelationship[] = [];
      let usingFallback = false;

      // Determine extraction strategy
      const method = this.config.method || 'pattern';
      const useApi = method !== 'pattern';
      
      // Try API extraction first if not in pattern-only mode
      if (useApi) {
        try {
          for (const source of sources) {
             // Skip empty content
             if (!source.content) continue;

             let apiResult: any;

             // If specifically requesting LLM/Hybrid and schema is present, try OneKE (Schema-guided)
             // Otherwise use DeepKE (General extraction)
             if ((method === 'llm' || method === 'hybrid') && this.config.extractionSchema && this.config.extractionSchema.length > 0) {
                // OneKE expects a single schema name or a predefined schema. 
                // Since extractionSchema is an array of strings, we'll join them or use the first as a hint
                // or default to a generic "UserDefined" schema name if the backend supports dynamic schemas.
                // For now, we'll pass the first schema item as the schema_name if valid, or default.
                const schemaName = this.config.extractionSchema[0] || 'Base';
                const oneKeResult = await knowledgeApi.extractOneKE({
                  text: source.content,
                  schema_name: schemaName
                });
                
                if (oneKeResult && oneKeResult.extracted_data) {
                   // Adapt OneKE result to internal format
                   // OneKE returns { entities: [], relations: [] } inside extracted_data
                   apiResult = oneKeResult.extracted_data;
                }
             } else {
                // Default to DeepKE
                apiResult = await knowledgeApi.extract({
                  text: source.content,
                  schema: this.config.extractionSchema
                });
             }

             if (apiResult) {
               // Map entities
               const sourceEntities: KnowledgeEntity[] = [];
               const threshold = this.config.confidenceThreshold || 0.6;

               if (apiResult.entities) {
                 apiResult.entities.forEach((e: any, idx: number) => {
                   const confidence = e.confidence || 0.8;
                   if (confidence >= threshold) {
                     sourceEntities.push({
                       id: `${source.id}-entity-${idx}-${Date.now()}`,
                       type: e.type || 'concept',
                       label: e.text || e.label,
                       properties: e.properties || {},
                       confidence: confidence,
                       source: source.id
                     });
                   }
                 });
               }
               entities.push(...sourceEntities);

               // Map relationships
               if (apiResult.relations) {
                 apiResult.relations.forEach((r: any, idx: number) => {
                   const confidence = r.confidence || 0.7;
                   if (confidence >= threshold) {
                     // Try to resolve head/tail to entity IDs
                     let sourceEntity = sourceEntities.find(e => e.label === r.head);
                     let targetEntity = sourceEntities.find(e => e.label === r.tail);

                     // Create missing entities if not found (cross-sentence or implied entities)
                     if (!sourceEntity) {
                       sourceEntity = {
                         id: `${source.id}-entity-${r.head}-${Date.now()}`,
                         type: 'concept',
                         label: r.head,
                         properties: {
                           sourceType: source.type,
                           implied: true
                         },
                         confidence: confidence * 0.9, // Slightly lower confidence for implied entities
                         source: source.id
                       };
                       sourceEntities.push(sourceEntity);
                       entities.push(sourceEntity);
                     }

                     if (!targetEntity) {
                       targetEntity = {
                         id: `${source.id}-entity-${r.tail}-${Date.now()}`,
                         type: 'concept',
                         label: r.tail,
                         properties: {
                           sourceType: source.type,
                           implied: true
                         },
                         confidence: confidence * 0.9, // Slightly lower confidence for implied entities
                         source: source.id
                       };
                       sourceEntities.push(targetEntity);
                       entities.push(targetEntity);
                     }

                     relationships.push({
                       id: `${source.id}-rel-${idx}-${Date.now()}`,
                       source: sourceEntity.id,
                       target: targetEntity.id,
                       type: r.relation,
                       properties: r.properties || {},
                       confidence: confidence
                     });
                   }
                 });
               }
             }
          }
        } catch (error) {
          console.warn('Knowledge Engine API failed, falling back to local pattern matching', error);
          usingFallback = true;
        }
      } else {
        usingFallback = true;
      }

      // If API yielded no results or failed, use fallback
      if (usingFallback || entities.length === 0) {
        context.updateProgress(50, 'Using fallback pattern extraction');
        entities = this.extractEntities(sources);
        relationships = this.extractRelationships(entities, sources);
      }

      context.updateProgress(90, 'Finalizing knowledge graph');

      // Update knowledge base if configured
      if (this.config.updateKnowledgeBase && entities.length > 0) {
        try {
          context.updateProgress(95, 'Updating knowledge base');
          await knowledgeApi.add({
            entities: entities.map(e => ({
              id: e.id,
              label: e.label,
              type: e.type,
              properties: { ...e.properties, confidence: e.confidence, source: e.source }
            })),
            relationships: relationships.map(r => ({
              source: r.source,
              target: r.target,
              type: r.type,
              properties: { ...r.properties, confidence: r.confidence }
            }))
          });
        } catch (error) {
          console.warn('Failed to update knowledge base:', error);
          // Don't fail the node execution, just warn
        }
      }

      const confidence = this.calculateConfidence(entities, relationships);
      const result: KnowledgeExtractionResult = {
        entities,
        relationships,
        confidence,
        metadata: {
          extractedAt: new Date(),
          executionTime: Date.now() - startTime,
          sourcesProcessed: sources.length,
        },
      };

      context.updateProgress(100, 'Knowledge extraction complete');
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during knowledge extraction'
      );
    }
  }

  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];
    const sources = inputs.sources || this.config.sources;

    if (!sources || !Array.isArray(sources) || sources.length === 0) {
      errors.push({
        field: 'sources',
        message: 'At least one source must be provided for extraction',
        severity: 'error',
      });
    }

    return errors;
  }

  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        method: {
          type: 'string',
          description: 'Extraction method',
          enum: ['pattern', 'ml', 'llm', 'hybrid'],
          default: 'pattern',
        },
        confidenceThreshold: {
          type: 'number',
          description: 'Minimum confidence threshold for extracted entities',
          minimum: 0,
          maximum: 1,
          default: 0.6,
        },
        useKnowledgeGraphs: {
          type: 'boolean',
          description: 'Enable knowledge graph enrichment',
          default: false,
        },
        extractionSchema: {
          type: 'array',
          description: 'Optional schema for extraction (e.g., ["entity", "relation"])',
          default: [],
        },
      },
      required: [],
    };
  }

  private extractEntities(sources: KnowledgeSource[]): KnowledgeEntity[] {
    const entities: KnowledgeEntity[] = [];
    const seen = new Set<string>();

    sources.forEach((source) => {
      const content = source.content || '';
      const matches = content.match(/\b[A-Z][a-zA-Z0-9_-]{2,}\b/g) || [];
      matches.forEach((match, index) => {
        const key = `${source.id}:${match}`;
        if (seen.has(key)) {
          return;
        }
        seen.add(key);
        entities.push({
          id: `${source.id}-entity-${index}`,
          type: 'concept',
          label: match,
          properties: {
            sourceType: source.type,
          },
          confidence: 0.7,
          source: source.id,
        });
      });
    });

    return entities;
  }

  private extractRelationships(entities: KnowledgeEntity[], sources: KnowledgeSource[]): KnowledgeRelationship[] {
    const relationships: KnowledgeRelationship[] = [];

    sources.forEach((source) => {
      const sourceEntities = entities.filter((entity) => entity.source === source.id);
      for (let i = 0; i < sourceEntities.length; i++) {
        for (let j = i + 1; j < sourceEntities.length; j++) {
          relationships.push({
            id: `${source.id}-rel-${i}-${j}`,
            source: sourceEntities[i].id,
            target: sourceEntities[j].id,
            type: 'co_occurs',
            properties: {
              source: source.id,
            },
            confidence: 0.5,
          });
        }
      }
    });

    return relationships;
  }

  private calculateConfidence(entities: KnowledgeEntity[], relationships: KnowledgeRelationship[]): number {
    if (entities.length === 0) {
      return 0;
    }

    const entityConfidence =
      entities.reduce((sum, entity) => sum + entity.confidence, 0) / entities.length;
    const relationshipConfidence = relationships.length
      ? relationships.reduce((sum, rel) => sum + rel.confidence, 0) / relationships.length
      : entityConfidence;

    return Number(((entityConfidence + relationshipConfidence) / 2).toFixed(2));
  }
}

export default KnowledgeExtractionNode;
