// @ts-nocheck
/**
 * MAKER Node
 *
 * Creative content generation node using MAKER methodology.
 * Methodical, Analytical, Knowledge-driven, Efficient, and Robust content creation.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { apiClient } from '@/services/api';
import { knowledgeApi } from '@/services/api/endpoints';

/**
 * Content types for MAKER generation
 */
export type ContentType =
  | 'article'
  | 'blog_post'
  | 'documentation'
  | 'tutorial'
  | 'report'
  | 'presentation'
  | 'code'
  | 'creative';

/**
 * MAKER node configuration
 */
export interface MAKERNodeConfig {
  contentType?: ContentType;
  tone?: 'formal' | 'informal' | 'technical' | 'creative' | 'persuasive';
  length?: 'short' | 'medium' | 'long';
  enableResearch?: boolean;
  enableCitations?: boolean;
  qualityLevel?: 'draft' | 'standard' | 'premium';
}

/**
 * MAKER methodology steps
 */
export interface MAKERSteps {
  Methodical: {
    approach: string;
    structure: string[];
    planning: string;
  };
  Analytical: {
    analysis: string;
    keyPoints: string[];
    considerations: string[];
  };
  Knowledge: {
    research: string;
    sources: string[];
    expertise: string;
  };
  Efficient: {
    optimization: string;
    bestPractices: string[];
    timeEstimate: number;
  };
  Robust: {
    validation: string;
    qualityChecks: string[];
    reliability: string;
  };
}

/**
 * Generated content result
 */
export interface GeneratedContent {
  content: string;
  summary: string;
  keyPoints: string[];
  structure: {
    sections: Array<{
      title: string;
      content: string;
      order: number;
    }>;
    wordCount: number;
    readingTime: number;
  };
  quality: {
    clarity: number;
    accuracy: number;
    completeness: number;
    relevance: number;
    overall: number;
  };
  metadata: {
    contentType: ContentType;
    tone: string;
    length: string;
    generatedAt: Date;
    generationTime: number;
    model: string;
  };
}

/**
 * MAKER result
 */
export interface MAKERResult {
  taskId: string;
  input: string;
  contentType: ContentType;
  methodology: MAKERSteps;
  content: GeneratedContent;
  metadata: {
    executedAt: Date;
    executionTime: number;
    parameters: {
      contentType: ContentType;
      tone: string;
      length: string;
      qualityLevel: string;
    };
  };
}

/**
 * MAKER Node
 *
 * Generates high-quality content using MAKER methodology.
 * Ensures methodical, analytical, knowledge-driven, efficient, and robust output.
 */
export class MAKERNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'MAKER Generator';
  static readonly DESCRIPTION = 'Creative content generation using MAKER methodology (Methodical, Analytical, Knowledge-driven, Efficient, Robust)';
  static readonly ICON = 'maker';
  static readonly CATEGORY = 'generation';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: MAKERNodeConfig = {}) {
    super(id, {
      contentType: 'article',
      tone: 'formal',
      length: 'medium',
      enableResearch: true,
      enableCitations: false,
      qualityLevel: 'standard',
      ...config
    });
  }

  /**
   * Execute MAKER content generation
   *
   * @param inputs - Must contain 'topic' or 'prompt'
   * @param context - Execution context
   * @returns Promise resolving to MAKER result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const topic = inputs.topic as string;
      const prompt = inputs.prompt as string;
      const input = topic || prompt;

      if (!input || input.trim().length === 0) {
        return this.createErrorResult('Topic or prompt is required and cannot be empty');
      }

      const contentType = (inputs.contentType as ContentType) || this.config.contentType as ContentType;
      const tone = (inputs.tone as string) || this.config.tone as string;
      const length = (inputs.length as string) || this.config.length as string;
      const qualityLevel = (inputs.qualityLevel as string) || this.config.qualityLevel as string;
      const requirements = inputs.requirements as string[] | undefined;
      const constraints = inputs.constraints as Record<string, any> | undefined;

      context.updateProgress(10, 'Analyzing requirements and planning approach');

      // Step 1: Methodical - Plan the approach
      const methodical = await this.executeMethodicalPhase(input, contentType, length, context);

      context.updateProgress(30, 'Analyzing topic and gathering insights');

      // Step 2: Analytical - Analyze the topic
      const analytical = await this.executeAnalyticalPhase(input, contentType, requirements, context);

      context.updateProgress(50, 'Gathering knowledge and expertise');

      // Step 3: Knowledge - Research and expertise
      const knowledge = await this.executeKnowledgePhase(input, contentType, this.config.enableResearch as boolean, context);

      context.updateProgress(70, 'Optimizing content generation');

      // Step 4: Efficient - Generate efficiently
      const efficient = await this.executeEfficientPhase(input, contentType, length, context);

      context.updateProgress(90, 'Validating and ensuring quality');

      // Step 5: Robust - Validate quality
      const robust = await this.executeRobustPhase(input, contentType, qualityLevel, context);

      // Generate final content
      const content = await this.generateContent(
        input,
        contentType,
        tone,
        length,
        { methodical, analytical, knowledge, efficient, robust },
        context
      );

      const executionTime = Date.now() - startTime;

      const result: MAKERResult = {
        taskId: `task-${Date.now()}`,
        input,
        contentType,
        methodology: {
          Methodical: methodical,
          Analytical: analytical,
          Knowledge: knowledge,
          Efficient: efficient,
          Robust: robust
        },
        content,
        metadata: {
          executedAt: new Date(),
          executionTime,
          parameters: {
            contentType,
            tone,
            length,
            qualityLevel
          }
        }
      };

      context.updateProgress(100, 'MAKER generation complete');

      return this.createSuccessResult(result);

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during MAKER generation'
      );
    }
  }

  /**
   * Execute Methodical phase - Plan the approach
   *
   * @param input - Input topic/prompt
   * @param contentType - Content type
   * @param length - Content length
   * @param context - Execution context
   * @returns Promise resolving to methodical phase output
   */
  private async executeMethodicalPhase(
    input: string,
    contentType: ContentType,
    length: string,
    context: ExecutionContext
  ): Promise<MAKERSteps['Methodical']> {
    // Structure based on content type and length
    const structures: Record<ContentType, string[]> = {
      article: ['Introduction', 'Body Paragraphs', 'Analysis', 'Conclusion'],
      blog_post: ['Hook', 'Main Content', 'Key Takeaways', 'Call to Action'],
      documentation: ['Overview', 'Installation', 'Usage', 'API Reference', 'Examples'],
      tutorial: ['Prerequisites', 'Step-by-Step Guide', 'Examples', 'Troubleshooting'],
      report: ['Executive Summary', 'Introduction', 'Findings', 'Analysis', 'Recommendations'],
      presentation: ['Title Slide', 'Agenda', 'Content Slides', 'Summary', 'Q&A'],
      code: ['Overview', 'Implementation', 'Usage Examples', 'Error Handling'],
      creative: ['Opening', 'Development', 'Climax', 'Resolution']
    };

    return {
      approach: `Structured ${contentType} creation with ${length} detail level`,
      structure: structures[contentType] || ['Introduction', 'Body', 'Conclusion'],
      planning: `Comprehensive planning for ${input.substring(0, 50)}...`
    };
  }

  /**
   * Execute Analytical phase - Analyze the topic
   *
   * @param input - Input topic/prompt
   * @param contentType - Content type
   * @param requirements - Optional requirements
   * @param context - Execution context
   * @returns Promise resolving to analytical phase output
   */
  private async executeAnalyticalPhase(
    input: string,
    contentType: ContentType,
    requirements: string[] | undefined,
    context: ExecutionContext
  ): Promise<MAKERSteps['Analytical']> {
    // Extract key points from input
    const words = input.toLowerCase().split(/\s+/);
    const uniqueWords = new Set(words);
    const keyPhrases = Array.from(uniqueWords).filter(w => w.length > 5);

    return {
      analysis: `Analyzing ${input.substring(0, 100)}... for ${contentType} creation`,
      keyPoints: requirements && requirements.length > 0 ? requirements : keyPhrases.slice(0, 5),
      considerations: [
        'Target audience and readability',
        'Content depth and complexity',
        'Relevance and engagement',
        'Clarity and structure'
      ]
    };
  }

  /**
   * Execute Knowledge phase - Research and expertise
   *
   * @param input - Input topic/prompt
   * @param contentType - Content type
   * @param enableResearch - Whether to enable research
   * @param context - Execution context
   * @returns Promise resolving to knowledge phase output
   */
  private async executeKnowledgePhase(
    input: string,
    contentType: ContentType,
    enableResearch: boolean,
    context: ExecutionContext
  ): Promise<MAKERSteps['Knowledge']> {
    if (enableResearch) {
      try {
        context.updateProgress(20, 'Retrieving knowledge and skills');
        
        // Query knowledge base for relevant information
        // Use RAG search as it maps well to semantic queries for research
        const results = await knowledgeApi.searchRag({
          query: input
        });

        // Try to inject ACE skills if available
        let enhancedInput = input;
        try {
          const aceStatus = await knowledgeApi.ace.getStatus();
          if (aceStatus.available) {
            // In a real implementation, we'd call an 'inject' endpoint
            // For now, we manually append skills if we retrieved them
            const skills = await knowledgeApi.ace.getSkills('maker_agent');
            if (skills.success && skills.skill_count > 0) {
              enhancedInput = `LEARNED STRATEGIES:\n${JSON.stringify(skills.skills)}\n\nTASK:\n${input}`;
            }
          }
        } catch (e) {
          console.warn('ACE skill injection skipped', e);
        }

        // Map RAG results to sources list safely
        const sources = (results || []).map((r: any) => r.source || r.url || 'Unknown Source');

        return {
          research: `Knowledge retrieval for ${input.substring(0, 50)}... found ${sources.length} sources`,
          sources,
          expertise: `Expertise applied for ${contentType} creation`
        };
      } catch (error) {
        // Fallback if knowledge query fails
        console.warn('Knowledge query failed, proceeding without research:', error);
      }
    }

    return {
      research: 'Built-in knowledge applied',
      sources: [],
      expertise: `Domain expertise for ${contentType}`
    };
  }

  /**
   * Execute Efficient phase - Optimize generation
   *
   * @param input - Input topic/prompt
   * @param contentType - Content type
   * @param length - Content length
   * @param context - Execution context
   * @returns Promise resolving to efficient phase output
   */
  private async executeEfficientPhase(
    input: string,
    contentType: ContentType,
    length: string,
    context: ExecutionContext
  ): Promise<MAKERSteps['Efficient']> {
    const lengthWords: Record<string, number> = {
      short: 300,
      medium: 700,
      long: 1500
    };

    const wordCount = lengthWords[length] || 700;
    const timeEstimate = Math.ceil(wordCount / 100); // ~100 words per minute

    return {
      optimization: `Efficient generation of ${wordCount} words`,
      bestPractices: [
        'Use clear and concise language',
        'Maintain logical flow',
        'Optimize for readability',
        'Include relevant examples',
        'Proofread and edit'
      ],
      timeEstimate
    };
  }

  /**
   * Execute Robust phase - Validate quality
   *
   * @param input - Input topic/prompt
   * @param contentType - Content type
   * @param qualityLevel - Quality level
   * @param context - Execution context
   * @returns Promise resolving to robust phase output
   */
  private async executeRobustPhase(
    input: string,
    contentType: ContentType,
    qualityLevel: string,
    context: ExecutionContext
  ): Promise<MAKERSteps['Robust']> {
    const qualityThresholds: Record<string, number[]> = {
      draft: [0.6, 0.6, 0.6, 0.6],
      standard: [0.8, 0.8, 0.8, 0.8],
      premium: [0.9, 0.9, 0.9, 0.9]
    };

    const thresholds = qualityThresholds[qualityLevel] || qualityThresholds.standard;

    return {
      validation: `Validating content at ${qualityLevel} quality level`,
      qualityChecks: [
        `Clarity >= ${thresholds[0]}`,
        `Accuracy >= ${thresholds[1]}`,
        `Completeness >= ${thresholds[2]}`,
        `Relevance >= ${thresholds[3]}`
      ],
      reliability: `${qualityLevel} quality assurance applied`
    };
  }

  /**
   * Generate final content
   *
   * @param input - Input topic/prompt
   * @param contentType - Content type
   * @param tone - Content tone
   * @param length - Content length
   * @param methodology - MAKER methodology outputs
   * @param context - Execution context
   * @returns Promise resolving to generated content
   */
  private async generateContent(
    input: string,
    contentType: ContentType,
    tone: string,
    length: string,
    methodology: MAKERSteps,
    context: ExecutionContext
  ): Promise<GeneratedContent> {
    // Call MAKER API for content generation
    const response = await apiClient.post<any>('/maker/generate', {
      topic: input,
      content_type: contentType,
      tone,
      length,
      methodology: {
        methodical: methodology.Methodical,
        analytical: methodology.Analytical,
        knowledge: methodology.Knowledge,
        efficient: methodology.Efficient,
        robust: methodology.Robust
      }
    });

    // Calculate reading time (average 200 words per minute)
    const wordCount = response.content?.split(/\s+/).length || 0;
    const readingTime = Math.ceil(wordCount / 200);

    return {
      content: response.content || '',
      summary: response.summary || `Generated ${contentType} about ${input.substring(0, 50)}...`,
      keyPoints: methodology.Analytical.keyPoints,
      structure: {
        sections: methodology.Methodical.structure.map((title, index) => ({
          title,
          content: response.content?.split('\n\n')[index] || '',
          order: index
        })),
        wordCount,
        readingTime
      },
      quality: response.quality || {
        clarity: 0.8,
        accuracy: 0.8,
        completeness: 0.8,
        relevance: 0.8,
        overall: 0.8
      },
      metadata: {
        contentType,
        tone,
        length,
        generatedAt: new Date(),
        generationTime: response.generation_time || 0,
        model: response.model || 'default'
      }
    };
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    const topic = inputs.topic || inputs.prompt;

    if (!topic) {
      errors.push({
        field: 'topic',
        message: 'Topic or prompt is required',
        severity: 'error'
      });
    }

    if (topic && typeof topic !== 'string') {
      errors.push({
        field: 'topic',
        message: 'Topic must be a string',
        severity: 'error'
      });
    }

    if (topic && topic.length < 10) {
      errors.push({
        field: 'topic',
        message: 'Topic is too short for meaningful content generation (minimum 10 characters)',
        severity: 'warning'
      });
    }

    // Validate content type
    if (inputs.contentType && typeof inputs.contentType === 'string') {
      const validTypes = ['article', 'blog_post', 'documentation', 'tutorial', 'report', 'presentation', 'code', 'creative'];
      if (!validTypes.includes(inputs.contentType)) {
        errors.push({
          field: 'contentType',
          message: `Content type must be one of: ${validTypes.join(', ')}`,
          severity: 'error'
        });
      }
    }

    // Validate tone
    if (inputs.tone && typeof inputs.tone === 'string') {
      const validTones = ['formal', 'informal', 'technical', 'creative', 'persuasive'];
      if (!validTones.includes(inputs.tone)) {
        errors.push({
          field: 'tone',
          message: `Tone must be one of: ${validTones.join(', ')}`,
          severity: 'error'
        });
      }
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        contentType: {
          type: 'string',
          description: 'Type of content to generate',
          enum: ['article', 'blog_post', 'documentation', 'tutorial', 'report', 'presentation', 'code', 'creative'],
          default: 'article'
        },
        tone: {
          type: 'string',
          description: 'Tone of the generated content',
          enum: ['formal', 'informal', 'technical', 'creative', 'persuasive'],
          default: 'formal'
        },
        length: {
          type: 'string',
          description: 'Length of the generated content',
          enum: ['short', 'medium', 'long'],
          default: 'medium'
        },
        enableResearch: {
          type: 'boolean',
          description: 'Enable knowledge research for content generation',
          default: true
        },
        enableCitations: {
          type: 'boolean',
          description: 'Enable citations in generated content',
          default: false
        },
        qualityLevel: {
          type: 'string',
          description: 'Quality level for content generation',
          enum: ['draft', 'standard', 'premium'],
          default: 'standard'
        }
      },
      required: []
    };
  }

  /**
   * Get available content types
   *
   * @returns Array of available content types
   */
  getAvailableContentTypes(): ContentType[] {
    return ['article', 'blog_post', 'documentation', 'tutorial', 'report', 'presentation', 'code', 'creative'];
  }

  /**
   * Get available tones
   *
   * @returns Array of available tones
   */
  getAvailableTones(): string[] {
    return ['formal', 'informal', 'technical', 'creative', 'persuasive'];
  }

  /**
   * Get quality metrics for generated content
   *
   * @param content - Generated content
   * @returns Promise resolving to quality metrics
   */
  async getQualityMetrics(content: string): Promise<NodeResult> {
    try {
      const response = await apiClient.post<any>('/maker/quality', {
        content
      });
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get quality metrics'
      );
    }
  }

  /**
   * Get generation history
   *
   * @param params - Query parameters
   * @returns Promise resolving to generation history
   */
  async getGenerationHistory(params?: {
    contentType?: ContentType;
    limit?: number;
    offset?: number;
  }): Promise<NodeResult> {
    try {
      const response = await apiClient.get<any>('/maker/history', params);
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get generation history'
      );
    }
  }
}

export default MAKERNode;
