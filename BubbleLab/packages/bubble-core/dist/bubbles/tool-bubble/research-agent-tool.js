import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { AIAgentBubble } from '../service-bubble/ai-agent.js';
import { AvailableModels } from '@bubblelab/shared-schemas';
import { parseJsonWithFallbacks } from '../../utils/json-parsing.js';
import { RECOMMENDED_MODELS } from '@bubblelab/shared-schemas';
import { isZodSchema, zodSchemaToJsonString } from '../../utils/zod-schema.js';
// Schema for the expected JSON result structure - accepts either a Zod schema or a JSON schema string
const ExpectedResultSchema = z.union([
    z.custom((val) => val?._def !== undefined),
    z.string(),
]);
// Define the parameters schema for the Research Agent Tool
const ResearchAgentToolParamsSchema = z.object({
    task: z
        .string()
        .min(1, 'Research task is required')
        .describe('The research task that requires searching the internet and gathering information'),
    expectedResultSchema: ExpectedResultSchema.describe('Zod schema or JSON schema string that defines the expected structure of the research result. Example: z.object({ trends: z.array(z.string()).describe("An array of trends"), summary: z.string().describe("A summary of the trends") }) or JSON.stringify({ type: "object", properties: { trends: { type: "array", items: { type: "string" } }, summary: { type: "string" } } })'),
    model: AvailableModels.describe(`Model to use for the research agent (default: ${RECOMMENDED_MODELS.BEST})`)
        .default(RECOMMENDED_MODELS.BEST)
        .describe(`Model to use for the research agent (default: ${RECOMMENDED_MODELS.BEST})`),
    maxTokens: z
        .number()
        .min(40000)
        .default(40000)
        .optional()
        .describe('Maximum number of tokens for the research agent (default: 40000)'),
    maxIterations: z
        .number()
        .min(1)
        .max(4000)
        .default(400)
        .describe('Maximum number of iterations for the research agent (default: 100)'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Required credentials'),
});
// Result schema for the research agent tool
const ResearchAgentToolResultSchema = z.object({
    result: z
        .any(z.unknown())
        .describe('The research result matching the expected JSON schema structure, parsed to object'),
    summary: z
        .string()
        .describe('1-2 sentence summary of what research was conducted and completed'),
    sourcesUsed: z
        .array(z.string())
        .describe('Array of URLs and sources that were searched and scraped during research'),
    iterationsUsed: z
        .number()
        .describe('Number of AI agent iterations used to complete the research'),
    success: z
        .boolean()
        .describe('Whether the research task was completed successfully'),
    error: z.string().describe('Error message if research failed'),
});
export class ResearchAgentTool extends ToolBubble {
    // Required static metadata
    static bubbleName = 'research-agent-tool';
    static schema = ResearchAgentToolParamsSchema;
    static resultSchema = ResearchAgentToolResultSchema;
    static shortDescription = 'AI-powered research agent that searches and scrapes the internet to gather structured information';
    static longDescription = `
    A sophisticated research agent that strategically combines web search and selective web scraping to gather and structure information from the internet.
    
    Features:
    - Intelligent web search using Firecrawl's search API to find relevant sources
    - Strategic web scraping - for detailed content from specific high-value pages
    - Multi-page web crawling - for comprehensive coverage across entire websites
    - AI-powered analysis to synthesize information into the requested JSON structure
    - Up to 100 iterations for thorough research and data gathering
    - Structured result formatting based on provided JSON schema
    - Comprehensive source tracking and research summary
    
    Research Strategy:
    - Prioritizes efficient web search to gather comprehensive information
    - Uses scraping strategically for detailed content from specific pages
    - Uses crawling for comprehensive coverage across multiple related pages
    - Only uses scraping/crawling when search results lack sufficient detail
    - Focuses on quality over quantity in content extraction
    
    Use cases:
    - Market research with structured competitor analysis
    - Academic research gathering from multiple sources  
    - Product research with feature comparisons
    - News and trend analysis with categorized findings
    - Technical research requiring documentation synthesis
    - Any research task requiring web data in a specific format
    
    The agent starts with systematic web searches, then strategically uses scraping for specific pages or crawling for comprehensive site coverage when additional detail is needed. It provides a summary of the research conducted and lists all sources used.
  `;
    static alias = 'research';
    static type = 'tool';
    constructor(params = {
        task: '',
        expectedResultSchema: z.object({ result: z.string() }),
    }, context) {
        super(params, context);
    }
    async performAction(context) {
        if (!this.params?.credentials?.[CredentialType.FIRECRAWL_API_KEY]) {
            return {
                result: {},
                summary: 'Research failed: FIRECRAWL_API_KEY is required',
                sourcesUsed: [],
                iterationsUsed: 0,
                success: false,
                error: 'FIRECRAWL_API_KEY is required',
            };
        }
        void context; // Context available but not currently used
        const { task, maxIterations } = this.params;
        const jsonSchemaString = this.getExpectedResultSchema();
        try {
            console.log('[ResearchAgentTool] Starting research task:', task.substring(0, 100) + '...');
            console.log('[ResearchAgentTool] Expected result schema (JSON):', jsonSchemaString.substring(0, 200) + '...');
            console.log('[ResearchAgentTool] Max iterations:', maxIterations);
            // Create the AI agent with web search and scraping tools
            const researchSubAgent = new AIAgentBubble({
                message: this.buildResearchPrompt(task, jsonSchemaString),
                systemPrompt: this.buildSystemPrompt(),
                model: {
                    model: this.params.model,
                    temperature: 1,
                    maxTokens: this.params.maxTokens,
                    jsonMode: true, // Enable JSON mode for structured output
                },
                tools: [
                    { name: 'web-search-tool' },
                    { name: 'web-scrape-tool' },
                    { name: 'web-crawl-tool' },
                    // { name: 'web-extract-tool' },
                    { name: 'reddit-scrape-tool' },
                ],
                maxIterations,
                credentials: this.params.credentials,
                streaming: false,
            }, this.context);
            console.log('[ResearchAgentTool] Executing AI agent...');
            const agentResult = await researchSubAgent.action();
            if (!agentResult.success) {
                throw new Error(`AI Agent failed: ${agentResult.error}`);
            }
            const agentData = agentResult.data;
            console.log('[ResearchAgentTool] AI agent completed successfully');
            console.log('[ResearchAgentTool] Iterations used:', agentData.iterations);
            console.log('[ResearchAgentTool] Tool calls made:', agentData.toolCalls.length);
            // Parse the AI agent's response as JSON with robust error handling
            let parsedResult;
            // Use the robust JSON parsing utilities that handle malformed JSON
            const parseResult = parseJsonWithFallbacks(agentData.response);
            if (!parseResult.success || parseResult.error) {
                // Check if this is already a processed error from the AI agent
                if (agentData.error &&
                    agentData.error.includes('failed to generate valid JSON')) {
                    throw new Error(`ResearchAgentTool failed: ${agentData.error}`);
                }
                // Use the robust parser's error message
                throw new Error(`ResearchAgentTool failed: AI Agent returned malformed JSON that could not be parsed: ${parseResult.error}. Response: ${agentData.response.substring(0, 200)}...`);
            }
            try {
                parsedResult = JSON.parse(parseResult.response);
            }
            catch (finalParseError) {
                // This should not happen with the robust parser, but just in case
                const originalError = finalParseError instanceof Error
                    ? finalParseError.message
                    : 'Unknown parsing error';
                throw new Error(`ResearchAgentTool failed: Final JSON parsing failed after robust processing: ${originalError}. Response: ${parseResult.response.substring(0, 200)}...`);
            }
            // Extract sources from tool calls
            const sourcesUsed = this.extractSourcesFromToolCalls(agentData.toolCalls);
            // Generate summary from the research process
            const summary = this.generateResearchSummary(task, agentData.toolCalls.length, sourcesUsed.length);
            console.log('[ResearchAgentTool] Research completed successfully');
            console.log('[ResearchAgentTool] Sources used:', sourcesUsed.length);
            console.log('[ResearchAgentTool] Summary:', summary);
            return {
                result: parsedResult,
                summary,
                sourcesUsed,
                iterationsUsed: agentData.iterations,
                success: true,
                error: '',
            };
        }
        catch (error) {
            console.error('[ResearchAgentTool] Research error:', error);
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            return {
                result: {},
                summary: `Research failed: ${errorMessage}`,
                sourcesUsed: [],
                iterationsUsed: 0,
                success: false,
                error: errorMessage,
            };
        }
    }
    getExpectedResultSchema() {
        // Use shared utility to convert Zod schema to JSON schema string
        if (isZodSchema(this.params.expectedResultSchema)) {
            return zodSchemaToJsonString(this.params.expectedResultSchema, 'ResultSchema');
        }
        return this.params.expectedResultSchema;
    }
    /**
     * Build the main research prompt for the AI agent
     */
    buildResearchPrompt(task, expectedResultSchema) {
        return `
Research Task: ${task}

Required Output Format (JSON Schema): ${expectedResultSchema}

Instructions:
1. Use web-search-tool to find relevant sources
2. Analyze the sources and choose the right tool:
   - If you need structured data extraction (images, prices, specific fields) → use web-extract-tool with a detailed prompt and schema
   - If multiple sources come from the same website → use web-crawl-tool for that site
   - If you're certain all needed info is on one specific page → use web-scrape-tool for that page
   - If scraping doesn't give complete results → use web-crawl-tool instead of more scraping
3. Never scrape multiple pages individually - always crawl the site instead
4. Return the final JSON result matching the expected schema

SPECIAL INSTRUCTIONS FOR IMAGE URL EXTRACTION:
- When extracting image URLs, look for the LARGEST, HIGHEST QUALITY product images
- Extract the DIRECT URL to the image file (ending in .jpg, .jpeg, .png, .webp, etc.)
- Avoid thumbnail or small preview images - find the main product gallery images
- Look for image URLs in src attributes, data-src attributes, or image CDN URLs
- Test that the URLs are accessible and point to actual image files
- If needed, convert relative URLs  to absolute URLs with the proper domain

SOURCE CREDIBILITY ASSESSMENT:
- Prioritize information from authoritative sources (.edu, .gov, reputable news outlets)
- Check for publication dates and prefer recent sources
- Cross-reference information across multiple sources
- Note any discrepancies or conflicts in information
- Assess source bias and reliability

CONTENT ANALYSIS TECHNIQUES:
- Extract key points and main arguments from sources
- Identify supporting evidence and data
- Note methodologies and data sources
- Extract quotes and statistics with attribution
- Synthesize information from multiple perspectives

CRITICAL: Return ONLY the final JSON result that matches the expected schema structure.

DO NOT include:
- Markdown code blocks with backticks
- Any explanatory text before or after the JSON
- Prefixes like "Here's the result:" or "The JSON is:"

Your response must start with { or [ and end with } or ] - nothing else.
    `.trim();
    }
    /**
     * Build the system prompt that defines the AI agent's behavior
     */
    buildSystemPrompt() {
        return `
You are a professional research agent specialized in gathering and structuring information from the internet. Your task is to use the following tools to gather information:

1. SEARCH SYSTEMATICALLY: Use web search to find the most relevant and authoritative sources
2. SCRAPE THOROUGHLY: Extract comprehensive information from discovered web pages, or when you are certain that the information you need is on one specific page, use the web-scrape-tool to scrape that page
3. CRAWL THOROUGHLY: Crawl the entire website to gather all the information you need if the scraping doesn't give complete results
4. RESEARCH RECURSIVELY: Continue searching and scraping until you have sufficient data
5. SYNTHESIZE INTELLIGENTLY: Organize all findings into the requested JSON structure

CRITICAL INSTRUCTIONS:
- DO NOT MAKE UP INFORMATION! All information must be from the sources you found.
- CITE YOUR SOURCES: Track which information came from which source
- ASSESS CREDIBILITY: Evaluate source quality and reliability
- CROSS-REFERENCE: Verify information across multiple sources
- EXTRACT KEY POINTS: Identify main arguments, evidence, and conclusions
- SYNTHESIZE: Combine information from multiple sources coherently

Research Guidelines:
- Start with web search to find relevant sources
- For structured data extraction (product info, images, prices) → use web-extract-tool with detailed prompts
- If search results show multiple pages from the same website → crawl that website
- If you're certain one page has all the info → scrape that specific page
- If scraping gives incomplete results → crawl the site instead of more scraping
- Never scrape multiple individual pages - always crawl the entire site instead

IMAGE EXTRACTION GUIDELINES:
- When extracting images, prioritize main product images over thumbnails
- Look for high-resolution images (usually larger file sizes, higher dimensions)
- Common image URL patterns: /images/, /media/, /assets/, CDN domains
- Check for lazy-loaded images in data-src, data-lazy-src attributes
- Ensure URLs are absolute (include https:// and domain)
- Validate that URLs end with image extensions (.jpg, .jpeg, .png, .webp, .gif)

CONTENT ANALYSIS METHODS:
1. KEY POINT EXTRACTION: Identify main ideas and arguments
2. EVIDENCE GATHERING: Extract supporting data and statistics
3. SOURCE ATTRIBUTION: Note where each piece of information comes from
4. CONTEXTUAL ANALYSIS: Understand the background and implications
5. CROSS-SOURCE SYNTHESIS: Combine information from multiple sources
6. IDENTIFYING GAPS: Note what information is missing or contradictory

NATURAL LANGUAGE PROCESSING TECHNIQUES:
- Extract relevant sentences and paragraphs
- Identify named entities (people, places, organizations)
- Extract relationships between concepts
- Summarize long content while preserving key information
- Detect sentiment and bias in sources
- Identify common themes across sources

SUMMARIZATION STRATEGIES:
EXTRACTIVE SUMMARIZATION:
- Select the most important sentences directly from source text
- Preserve original wording and attribution
- Maintain factual accuracy
- Use lead sentences and topic sentences

ABSTRACTIVE SUMMARIZATION:
- Paraphrase and condense information
- Combine related concepts
- Maintain key meaning while reducing length
- Add connecting phrases for coherence

Output Requirements:
- Return ONLY valid JSON that matches the provided schema
- NO markdown code blocks, explanations, or additional text
- NO prefixes like "Here's the result:" or "The JSON is:"
- Start your response directly with { or [ (the JSON structure)
- End your response with } or ] (closing the JSON structure)
- Ensure all required schema fields are addressed, if not available leave empty
- Include all relevant information you discovered
- Maintain data accuracy and cite reliable sources in your research process
- Organize information logically with clear structure
- Use consistent formatting and terminology

JSON FORMATTING RULES:
- Use double quotes for all strings and property names
- No trailing commas
- No single quotes
- No unescaped newlines in strings
- Properly escape special characters in strings

You have access to web-search-tool, web-scrape-tool, web-crawl-tool, and web-extract-tool. Use web-extract-tool for structured data extraction (like product images, prices, specific fields). Prefer web-crawl-tool over web-scrape-tool when you need information from multiple pages of a website.
    `.trim();
    }
    /**
     * Extract URLs and sources from the tool calls made during research
     * Enhanced with source credibility scoring and categorization
     */
    extractSourcesFromToolCalls(toolCalls) {
        const sources = [];
        for (const toolCall of toolCalls) {
            if (toolCall.tool === 'web-search-tool' && toolCall.output) {
                // Extract URLs from search results with credibility scoring
                const output = toolCall.output;
                if (output.results) {
                    for (const result of output.results) {
                        if (result.url) {
                            // Score source credibility
                            const credibilityScore = this.scoreSourceCredibility(result.url);
                            console.log(`[ResearchAgentTool] Source ${result.url} - Credibility: ${credibilityScore}`);
                            sources.push(result.url);
                        }
                    }
                }
            }
            else if (toolCall.tool === 'web-scrape-tool' && toolCall.input) {
                // Extract URL from scrape input
                const input = toolCall.input;
                if (input.url) {
                    sources.push(input.url);
                }
            }
            else if (toolCall.tool === 'web-crawl-tool' && toolCall.input) {
                // Extract URL from crawl input
                const input = toolCall.input;
                if (input.url) {
                    sources.push(input.url);
                }
                // Also extract URLs from crawl output if available
                if (toolCall.output) {
                    const output = toolCall.output;
                    if (output.pages) {
                        for (const page of output.pages) {
                            if (page.url) {
                                sources.push(page.url);
                            }
                        }
                    }
                }
            }
            else if (toolCall.tool === 'web-extract-tool' && toolCall.input) {
                // Extract URL from extract input
                const input = toolCall.input;
                if (input.url) {
                    sources.push(input.url);
                }
            }
        }
        // Remove duplicates and return
        return [...new Set(sources)];
    }
    /**
     * Score source credibility based on URL characteristics
     * Returns a score from 0 (low credibility) to 1 (high credibility)
     */
    scoreSourceCredibility(url) {
        let score = 0.5; // Base score
        try {
            const urlObj = new URL(url);
            const domain = urlObj.hostname.toLowerCase();
            // High credibility domains
            if (domain.endsWith('.edu') ||
                domain.endsWith('.gov') ||
                domain.includes('.gov.') ||
                domain.includes('scholar.') ||
                domain.includes('research.')) {
                score = 0.95;
            }
            // Reputable news outlets
            else if (domain.includes('reuters.com') ||
                domain.includes('apnews.com') ||
                domain.includes('bbc.com') ||
                domain.includes('npr.org') ||
                domain.includes('nytimes.com') ||
                domain.includes('washingtonpost.com') ||
                domain.includes('wsj.com') ||
                domain.includes('economist.com')) {
                score = 0.90;
            }
            // Reputable academic sources
            else if (domain.includes('ieee.org') ||
                domain.includes('acm.org') ||
                domain.includes('nature.com') ||
                domain.includes('science.org') ||
                domain.includes('springer.com') ||
                domain.includes('sciencedirect.com') ||
                domain.includes('jstor.org') ||
                domain.includes('pubmed.ncbi.nlm.nih.gov')) {
                score = 0.92;
            }
            // Reputable tech sources
            else if (domain.includes('w3.org') ||
                domain.includes('developer.mozilla.org') ||
                domain.includes('docs.microsoft.com') ||
                domain.includes('devdocs.io')) {
                score = 0.88;
            }
            // Blogs and personal sites (lower credibility)
            else if (domain.includes('blog') ||
                domain.includes('wordpress.com') ||
                domain.includes('blogspot.com') ||
                domain.includes('medium.com') ||
                domain.includes('substack.com')) {
                score = 0.60;
            }
            // Social media (very low credibility for factual info)
            else if (domain.includes('twitter.com') ||
                domain.includes('facebook.com') ||
                domain.includes('instagram.com') ||
                domain.includes('tiktok.com') ||
                domain.includes('reddit.com')) {
                score = 0.40;
            }
            // Content farms and low-quality sources
            else if (domain.includes('wikihow.com') ||
                domain.includes('about.com') ||
                domain.includes('answers.com')) {
                score = 0.50;
            }
            // HTTPS indicates some level of security
            if (url.startsWith('https://')) {
                score += 0.05;
            }
            // Penalize very new domains (less than 6 months old) - rough heuristic
            // This is not perfect but provides some signal
            // In production, you'd use a domain age API
            // Penalize URL patterns associated with low-quality content
            if (url.includes('/blog/') ||
                url.includes('/opinion/') ||
                url.includes('/commentary/')) {
                score -= 0.10;
            }
            // Boost for URL patterns associated with high-quality content
            if (url.includes('/research/') ||
                url.includes('/publications/') ||
                url.includes('/journals/') ||
                url.includes('/docs/') ||
                url.includes('/documentation/')) {
                score += 0.05;
            }
        }
        catch (error) {
            // Invalid URL, keep base score
            console.warn(`[ResearchAgentTool] Invalid URL for credibility scoring: ${url}`);
        }
        // Clamp score to [0, 1]
        return Math.max(0, Math.min(1, score));
    }
    /**
     * Extract key points from research content
     * Uses NLP techniques to identify important sentences and concepts
     */
    extractKeyPoints(content, maxPoints = 5) {
        // Split content into sentences
        const sentences = content
            .split(/[.!?]+/)
            .map((s) => s.trim())
            .filter((s) => s.length > 20);
        // Simple scoring based on sentence features
        const scored = sentences.map((sentence) => {
            let score = 0;
            // Longer sentences tend to be more informative
            score += Math.min(sentence.length / 100, 1);
            // Sentences with numbers often contain data
            if (/\d+/.test(sentence)) {
                score += 0.5;
            }
            // Sentences with quotes contain evidence
            if (sentence.includes('"') || sentence.includes("'")) {
                score += 0.3;
            }
            // Sentences at the beginning are often introductory
            const index = sentences.indexOf(sentence);
            if (index < 3) {
                score += 0.4;
            }
            // Sentences with certain keywords are important
            const importantWords = [
                'conclusion',
                'result',
                'finding',
                'study',
                'research',
                'showed',
                'demonstrated',
                'significant',
                'important',
                'key',
                'main',
                'primary',
            ];
            const lowerSentence = sentence.toLowerCase();
            importantWords.forEach((word) => {
                if (lowerSentence.includes(word)) {
                    score += 0.2;
                }
            });
            return { sentence, score };
        });
        // Sort by score and return top N
        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, maxPoints).map((item) => item.sentence);
    }
    /**
     * Summarize research content
     * Combines extractive and abstractive summarization techniques
     */
    summarizeContent(content, maxLength = 500) {
        const sentences = content
            .split(/[.!?]+/)
            .map((s) => s.trim())
            .filter((s) => s.length > 10);
        if (sentences.length === 0) {
            return '';
        }
        // Extractive: Use first and last sentences (often contain intro and conclusion)
        const firstSentence = sentences[0];
        const lastSentence = sentences[sentences.length - 1];
        // Add middle sentences with key points
        const middleSentences = sentences.slice(1, -1);
        const keyPoints = this.extractKeyPoints(middleSentences.join(' '), 3);
        // Combine and truncate
        let summary = `${firstSentence}. ${keyPoints.join(' ')} ${lastSentence}.`;
        if (summary.length > maxLength) {
            // Truncate at last complete sentence under limit
            const truncated = summary.substring(0, maxLength);
            const lastPeriod = truncated.lastIndexOf('.');
            if (lastPeriod > 0) {
                summary = truncated.substring(0, lastPeriod + 1);
            }
            else {
                summary = truncated + '...';
            }
        }
        return summary;
    }
    /**
     * Generate a concise summary of the research conducted
     */
    generateResearchSummary(task, toolCallsCount, sourcesCount) {
        const taskPreview = task.length > 50 ? task.substring(0, 50) + '...' : task;
        return `Completed research on "${taskPreview}" using ${toolCallsCount} tool operations across ${sourcesCount} web sources. Gathered and structured information according to the requested schema format.`;
    }
}
//# sourceMappingURL=research-agent-tool.js.map