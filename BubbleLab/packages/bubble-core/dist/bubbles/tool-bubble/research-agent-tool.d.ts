import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const ResearchAgentToolParamsSchema: z.ZodObject<{
    task: z.ZodString;
    expectedResultSchema: z.ZodUnion<[z.ZodType<z.ZodTypeAny, z.ZodTypeDef, z.ZodTypeAny>, z.ZodString]>;
    model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
    maxTokens: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    maxIterations: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
    maxIterations: number;
    task: string;
    expectedResultSchema: string | z.ZodTypeAny;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxTokens?: number | undefined;
}, {
    task: string;
    expectedResultSchema: string | z.ZodTypeAny;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
    maxTokens?: number | undefined;
    maxIterations?: number | undefined;
}>;
declare const ResearchAgentToolResultSchema: z.ZodObject<{
    result: z.ZodAny;
    summary: z.ZodString;
    sourcesUsed: z.ZodArray<z.ZodString, "many">;
    iterationsUsed: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    summary: string;
    sourcesUsed: string[];
    iterationsUsed: number;
    result?: any;
}, {
    error: string;
    success: boolean;
    summary: string;
    sourcesUsed: string[];
    iterationsUsed: number;
    result?: any;
}>;
type ResearchAgentToolParams = z.output<typeof ResearchAgentToolParamsSchema>;
type ResearchAgentToolResult = z.output<typeof ResearchAgentToolResultSchema>;
type ResearchAgentToolParamsInput = z.input<typeof ResearchAgentToolParamsSchema>;
export declare class ResearchAgentTool extends ToolBubble<ResearchAgentToolParams, ResearchAgentToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        task: z.ZodString;
        expectedResultSchema: z.ZodUnion<[z.ZodType<z.ZodTypeAny, z.ZodTypeDef, z.ZodTypeAny>, z.ZodString]>;
        model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
        maxTokens: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        maxIterations: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        maxIterations: number;
        task: string;
        expectedResultSchema: string | z.ZodTypeAny;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxTokens?: number | undefined;
    }, {
        task: string;
        expectedResultSchema: string | z.ZodTypeAny;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        maxTokens?: number | undefined;
        maxIterations?: number | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        result: z.ZodAny;
        summary: z.ZodString;
        sourcesUsed: z.ZodArray<z.ZodString, "many">;
        iterationsUsed: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        summary: string;
        sourcesUsed: string[];
        iterationsUsed: number;
        result?: any;
    }, {
        error: string;
        success: boolean;
        summary: string;
        sourcesUsed: string[];
        iterationsUsed: number;
        result?: any;
    }>;
    static readonly shortDescription = "AI-powered research agent that searches and scrapes the internet to gather structured information";
    static readonly longDescription = "\n    A sophisticated research agent that strategically combines web search and selective web scraping to gather and structure information from the internet.\n    \n    Features:\n    - Intelligent web search using Firecrawl's search API to find relevant sources\n    - Strategic web scraping - for detailed content from specific high-value pages\n    - Multi-page web crawling - for comprehensive coverage across entire websites\n    - AI-powered analysis to synthesize information into the requested JSON structure\n    - Up to 100 iterations for thorough research and data gathering\n    - Structured result formatting based on provided JSON schema\n    - Comprehensive source tracking and research summary\n    \n    Research Strategy:\n    - Prioritizes efficient web search to gather comprehensive information\n    - Uses scraping strategically for detailed content from specific pages\n    - Uses crawling for comprehensive coverage across multiple related pages\n    - Only uses scraping/crawling when search results lack sufficient detail\n    - Focuses on quality over quantity in content extraction\n    \n    Use cases:\n    - Market research with structured competitor analysis\n    - Academic research gathering from multiple sources  \n    - Product research with feature comparisons\n    - News and trend analysis with categorized findings\n    - Technical research requiring documentation synthesis\n    - Any research task requiring web data in a specific format\n    \n    The agent starts with systematic web searches, then strategically uses scraping for specific pages or crawling for comprehensive site coverage when additional detail is needed. It provides a summary of the research conducted and lists all sources used.\n  ";
    static readonly alias = "research";
    static readonly type = "tool";
    constructor(params?: ResearchAgentToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<ResearchAgentToolResult>;
    getExpectedResultSchema(): string;
    /**
     * Build the main research prompt for the AI agent
     */
    private buildResearchPrompt;
    /**
     * Build the system prompt that defines the AI agent's behavior
     */
    private buildSystemPrompt;
    /**
     * Extract URLs and sources from the tool calls made during research
     * Enhanced with source credibility scoring and categorization
     */
    private extractSourcesFromToolCalls;
    /**
     * Score source credibility based on URL characteristics
     * Returns a score from 0 (low credibility) to 1 (high credibility)
     */
    private scoreSourceCredibility;
    /**
     * Extract key points from research content
     * Uses NLP techniques to identify important sentences and concepts
     */
    private extractKeyPoints;
    /**
     * Summarize research content
     * Combines extractive and abstractive summarization techniques
     */
    private summarizeContent;
    /**
     * Generate a concise summary of the research conducted
     */
    private generateResearchSummary;
}
export {};
//# sourceMappingURL=research-agent-tool.d.ts.map