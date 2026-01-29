import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { inventionApi, knowledgeApi } from '../services/api/endpoints';

/**
 * Invention domains
 */
export type InventionDomain =
  | 'technology'      // Technology/Software
  | 'hardware'        // Hardware/Physical
  | 'business'        // Business Model
  | 'process'         // Process/System
  | 'scientific'      // Scientific Discovery
  | 'creative';       // Creative/Artistic

/**
 * Planning stages
 */
export type PlanningStage =
  | 'research'        // Research phase
  | 'ideation'        // Ideation
  | 'prototyping'     // Prototyping
  | 'testing'         // Testing
  | 'validation'      // Validation
  | 'scaling'         // Scaling
  | 'commercialization'; // Commercialization

/**
 * Detail levels
 */
export type DetailLevel =
  | 'overview'        // High-level overview
  | 'detailed'        // Detailed plan
  | 'comprehensive';  // All details

/**
 * Invention planning result
 */
export interface InventionResult {
  plan: any;
  priorArt?: any;
  feasibility?: any;
  roadmap?: any;
  leanProofs?: any[];
  errorAnalysis: any;
  redTeamResults?: any;
  blueTeamResults?: any;
  successCriteria: any[];
  executionTime: number;
  qualityAssessment: {
    innovation: number;
    feasibility: number;
    clarity: number;
    completeness: number;
  };
}

/**
 * Invention node configuration
 */
export interface InventionNodeConfig {
  goal?: string;
  domain?: InventionDomain;
  innovativeness?: number;
  planningStages?: PlanningStage[];
  constraints?: string;
  targetAudience?: string;
  includePriorArt?: boolean;
  includeFeasibility?: boolean;
  includeRoadmap?: boolean;
  detailLevel?: DetailLevel;
}

/**
 * Invention Planner Node class
 */
export class InventionNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'End-to-End Invention Planner';
  static readonly DESCRIPTION = 'Systematic invention planning and development via ASR-GoT framework';
  static readonly ICON = '💡';
  static readonly CATEGORY = 'planning';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: InventionNodeConfig = {}) {
    super(id, {
      goal: '',
      domain: 'technology',
      innovativeness: 0.7,
      planningStages: ['research', 'ideation', 'prototyping'],
      includePriorArt: true,
      includeFeasibility: true,
      includeRoadmap: true,
      detailLevel: 'detailed',
      ...config
    });
  }

  /**
   * Execute invention planning
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();
      const goal = (inputs.goal as string) || (this.config.goal as string);
      const domain = (inputs.domain as InventionDomain) || (this.config.domain as InventionDomain);
      
      if (!goal || goal.trim().length === 0) {
        return this.createErrorResult('Invention goal is required');
      }

      context.updateProgress(10, 'Analyzing goal and gathering context');

      // Step 1: Knowledge Retrieval (if configured)
      let priorArtContext = '';
      if (this.config.includePriorArt) {
        try {
          context.updateProgress(20, 'Searching for prior art and related knowledge');
          const searchResults = await knowledgeApi.searchRag({ query: goal });
          if (searchResults && searchResults.length > 0) {
            priorArtContext = searchResults.map(r => r.content).join('\n\n');
          }
        } catch (error) {
          console.warn('Knowledge search for invention failed, proceeding with goal only', error);
        }
      }

      context.updateProgress(40, 'Generating end-to-end invention plan');

      // Step 2: Call Invention API
      const result = await inventionApi.createPlan({
        goal: goal + (priorArtContext ? `\n\nContext:\n${priorArtContext}` : ''),
        domain,
        innovativeness: this.config.innovativeness || 0.7,
        planning_stages: this.config.planningStages || [],
        constraints: (inputs.constraints as string) || this.config.constraints,
        target_audience: (inputs.targetAudience as string) || this.config.targetAudience,
        include_prior_art: this.config.includePriorArt || false,
        include_feasibility: this.config.includeFeasibility || false,
        include_roadmap: this.config.includeRoadmap || false,
        detail_level: this.config.detailLevel || 'detailed'
      });

      const executionTime = (Date.now() - startTime) / 1000;

      context.updateProgress(100, 'Invention plan completed successfully');

      return this.createSuccessResult({
        ...result,
        executionTime
      });

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during invention planning'
      );
    }
  }

  /**
   * Validate inputs
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];
    const goal = inputs.goal || this.config.goal;

    if (!goal || goal.trim().length === 0) {
      errors.push({
        field: 'goal',
        message: 'Invention goal is required',
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Get parameter schema
   */
  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        goal: {
          type: 'string',
          description: 'What do you want to invent?',
          required: true
        },
        domain: {
          type: 'string',
          description: 'Primary domain of invention',
          enum: ['technology', 'hardware', 'business', 'process', 'scientific', 'creative'],
          default: 'technology'
        },
        innovativeness: {
          type: 'number',
          description: 'radicalness level (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.7
        },
        includePriorArt: {
          type: 'boolean',
          description: 'Include knowledge-based prior art analysis',
          default: true
        },
        includeFeasibility: {
          type: 'boolean',
          description: 'Include feasibility analysis',
          default: true
        },
        includeRoadmap: {
          type: 'boolean',
          description: 'Include implementation roadmap',
          default: true
        },
        detailLevel: {
          type: 'string',
          description: 'Level of detail in the generated plan',
          enum: ['overview', 'detailed', 'comprehensive'],
          default: 'detailed'
        }
      },
      required: []
    };
  }
}

export default InventionNode;
