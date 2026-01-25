import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Content types for MAKER generation
 */
export type ContentType = 'article' | 'blog_post' | 'documentation' | 'tutorial' | 'report' | 'presentation' | 'code' | 'creative';
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
export declare class MAKERNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "MAKER Generator";
    static readonly DESCRIPTION = "Creative content generation using MAKER methodology (Methodical, Analytical, Knowledge-driven, Efficient, Robust)";
    static readonly ICON = "maker";
    static readonly CATEGORY = "generation";
    static readonly VERSION = "1.0.0";
    constructor(id: string, config?: MAKERNodeConfig);
    /**
     * Execute MAKER content generation
     *
     * @param inputs - Must contain 'topic' or 'prompt'
     * @param context - Execution context
     * @returns Promise resolving to MAKER result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Execute Methodical phase - Plan the approach
     *
     * @param input - Input topic/prompt
     * @param contentType - Content type
     * @param length - Content length
     * @param context - Execution context
     * @returns Promise resolving to methodical phase output
     */
    private executeMethodicalPhase;
    /**
     * Execute Analytical phase - Analyze the topic
     *
     * @param input - Input topic/prompt
     * @param contentType - Content type
     * @param requirements - Optional requirements
     * @param context - Execution context
     * @returns Promise resolving to analytical phase output
     */
    private executeAnalyticalPhase;
    /**
     * Execute Knowledge phase - Research and expertise
     *
     * @param input - Input topic/prompt
     * @param contentType - Content type
     * @param enableResearch - Whether to enable research
     * @param context - Execution context
     * @returns Promise resolving to knowledge phase output
     */
    private executeKnowledgePhase;
    /**
     * Execute Efficient phase - Optimize generation
     *
     * @param input - Input topic/prompt
     * @param contentType - Content type
     * @param length - Content length
     * @param context - Execution context
     * @returns Promise resolving to efficient phase output
     */
    private executeEfficientPhase;
    /**
     * Execute Robust phase - Validate quality
     *
     * @param input - Input topic/prompt
     * @param contentType - Content type
     * @param qualityLevel - Quality level
     * @param context - Execution context
     * @returns Promise resolving to robust phase output
     */
    private executeRobustPhase;
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
    private generateContent;
    /**
     * Validate input data
     *
     * @param inputs - Input data to validate
     * @returns Array of validation errors
     */
    validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get JSON Schema for configuration parameters
     *
     * @returns Parameter schema
     */
    getParameterSchema(): ParameterSchema;
    /**
     * Get available content types
     *
     * @returns Array of available content types
     */
    getAvailableContentTypes(): ContentType[];
    /**
     * Get available tones
     *
     * @returns Array of available tones
     */
    getAvailableTones(): string[];
    /**
     * Get quality metrics for generated content
     *
     * @param content - Generated content
     * @returns Promise resolving to quality metrics
     */
    getQualityMetrics(content: string): Promise<NodeResult>;
    /**
     * Get generation history
     *
     * @param params - Query parameters
     * @returns Promise resolving to generation history
     */
    getGenerationHistory(params?: {
        contentType?: ContentType;
        limit?: number;
        offset?: number;
    }): Promise<NodeResult>;
}
export default MAKERNode;
