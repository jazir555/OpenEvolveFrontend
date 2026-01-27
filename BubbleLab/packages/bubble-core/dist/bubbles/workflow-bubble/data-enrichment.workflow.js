/**
 * DATA ENRICHMENT WORKFLOW
 *
 * A comprehensive workflow for enriching data records from multiple sources
 * including web search, vector search, and AI analysis.
 *
 * This workflow combines:
 * 1. Web search tool for external data enrichment
 * 2. Vector search for similar record discovery
 * 3. AI agent for intelligent analysis and synthesis
 * 4. Multi-source data merging and validation
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import { CredentialType, AvailableModels } from '@bubblelab/shared-schemas';
import { AIAgentBubble } from '../service-bubble/ai-agent.js';
import { HttpBubble } from '../service-bubble/http.js';
/**
 * Data enrichment source configuration
 */
const EnrichmentSourceSchema = z.object({
    webSearch: z
        .boolean()
        .default(false)
        .describe('Enable web search enrichment'),
    vectorSearch: z
        .boolean()
        .default(false)
        .describe('Enable vector similarity search'),
    aiAnalysis: z
        .boolean()
        .default(true)
        .describe('Enable AI-powered analysis'),
    databaseLookup: z
        .boolean()
        .default(false)
        .describe('Enable database lookup enrichment'),
});
/**
 * Parameters schema for data enrichment workflow
 */
const DataEnrichmentParamsSchema = z.object({
    /**
     * Input record to enrich
     */
    record: z
        .record(z.unknown())
        .describe('Input data record to enrich'),
    /**
     * Enrichment sources to use
     */
    sources: z
        .object({
        webSearch: z.boolean().default(false),
        vectorSearch: z.boolean().default(false),
        aiAnalysis: z.boolean().default(true),
        databaseLookup: z.boolean().default(false),
    })
        .optional()
        .describe('Which enrichment sources to use'),
    /**
     * Web search configuration
     */
    webSearchConfig: z
        .object({
        searchEngine: z
            .enum(['google', 'bing', 'duckduckgo'])
            .default('google')
            .describe('Search engine to use'),
        maxResults: z
            .number()
            .int()
            .positive()
            .default(5)
            .describe('Maximum number of search results'),
        searchQuery: z
            .string()
            .optional()
            .describe('Custom search query (defaults to record-based query)'),
    })
        .optional()
        .describe('Web search configuration'),
    /**
     * Vector search configuration
     */
    vectorSearchConfig: z
        .object({
        vectorEndpoint: z
            .string()
            .url()
            .describe('Vector search API endpoint'),
        topK: z
            .number()
            .int()
            .positive()
            .default(5)
            .describe('Number of similar records to retrieve'),
        threshold: z
            .number()
            .min(0)
            .max(1)
            .default(0.7)
            .describe('Similarity threshold'),
    })
        .optional()
        .describe('Vector search configuration'),
    /**
     * Database lookup configuration
     */
    databaseConfig: z
        .object({
        connectionString: z
            .string()
            .describe('Database connection string'),
        query: z
            .string()
            .describe('SQL query for enrichment'),
        queryKey: z
            .string()
            .describe('Key field from record to use in query'),
    })
        .optional()
        .describe('Database lookup configuration'),
    /**
     * AI analysis configuration
     */
    aiConfig: z
        .object({
        model: AvailableModels.default('google/gemini-2.5-flash'),
        temperature: z
            .number()
            .min(0)
            .max(2)
            .default(0.3)
            .describe('AI model temperature'),
        maxTokens: z
            .number()
            .positive()
            .default(50000)
            .describe('Maximum AI response tokens'),
        analysisPrompt: z
            .string()
            .optional()
            .describe('Custom AI analysis prompt'),
    })
        .optional()
        .describe('AI analysis configuration'),
    /**
     * Output format
     */
    outputFormat: z
        .enum(['merged', 'append', 'replace'])
        .default('merged')
        .describe('How to combine enriched data with original record'),
    /**
     * Credentials
     */
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Credentials for external services'),
});
/**
 * Result schema for data enrichment workflow
 */
const DataEnrichmentResultSchema = z.object({
    success: z.boolean(),
    error: z.string(),
    /**
     * Enriched record
     */
    enrichedRecord: z
        .record(z.unknown())
        .optional()
        .describe('Enriched data record'),
    /**
     * Enrichment sources results
     */
    enrichmentResults: z
        .object({
        webSearch: z
            .object({
            success: z.boolean(),
            results: z.array(z.unknown()).optional(),
            count: z.number().optional(),
        })
            .optional(),
        vectorSearch: z
            .object({
            success: z.boolean(),
            similarRecords: z.array(z.unknown()).optional(),
            count: z.number().optional(),
        })
            .optional(),
        aiAnalysis: z
            .object({
            success: z.boolean(),
            insights: z.string().optional(),
            confidence: z.number().optional(),
        })
            .optional(),
        databaseLookup: z
            .object({
            success: z.boolean(),
            data: z.unknown().optional(),
            rowsAffected: z.number().optional(),
        })
            .optional(),
    })
        .optional()
        .describe('Results from each enrichment source'),
    /**
     * Enrichment metadata
     */
    metadata: z
        .object({
        sourcesUsed: z.array(z.string()),
        enrichmentTimestamp: z.date(),
        processingTime: z.number(),
        fieldsAdded: z.number(),
        dataQualityScore: z.number(),
    })
        .optional(),
});
/**
 * Data Enrichment Workflow
 *
 * Enriches data records from multiple sources with intelligent merging and validation.
 */
export class DataEnrichmentWorkflow extends WorkflowBubble {
    static type = 'workflow';
    static bubbleName = 'data-enrichment-workflow';
    static schema = DataEnrichmentParamsSchema;
    static resultSchema = DataEnrichmentResultSchema;
    static shortDescription = 'Multi-source data enrichment with AI-powered analysis';
    static longDescription = `
    Enriches data records by combining multiple data sources and AI analysis.

    Features:
    - Web search for external information retrieval
    - Vector similarity search for related records
    - Database lookup for structured data enrichment
    - AI-powered analysis and synthesis
    - Intelligent data merging strategies
    - Data quality scoring and validation

    Use cases:
    - CRM record enrichment with external data
    - Lead scoring with additional context
    - Product data enhancement from multiple sources
    - Customer profile enrichment
    - Research data augmentation

    Process:
    1. Extract key information from input record
    2. Query enabled enrichment sources in parallel
    3. AI analyzes and synthesizes all gathered data
    4. Merge enriched data with original record
    5. Calculate data quality score
    6. Return comprehensive enrichment results
  `;
    static alias = 'enrich-data';
    constructor(params, context) {
        super(params, context);
    }
    async performAction() {
        const startTime = Date.now();
        console.log('[DataEnrichment] Starting data enrichment workflow');
        console.log('[DataEnrichment] Input record keys:', Object.keys(this.params.record));
        const sources = this.params.sources || {};
        const enrichmentResults = {};
        const enrichedRecord = { ...this.params.record };
        const sourcesUsed = [];
        try {
            // Step 1: Web Search Enrichment
            if (sources.webSearch) {
                console.log('[DataEnrichment] Step 1: Web search enrichment');
                sourcesUsed.push('webSearch');
                const webSearchResult = await this.performWebSearch();
                enrichmentResults.webSearch = {
                    success: webSearchResult.success,
                    results: webSearchResult.data,
                    count: webSearchResult.data?.length,
                };
                if (webSearchResult.success && webSearchResult.data) {
                    enrichedRecord.webSearchResults = webSearchResult.data;
                }
            }
            // Step 2: Vector Search Enrichment
            if (sources.vectorSearch) {
                console.log('[DataEnrichment] Step 2: Vector search enrichment');
                sourcesUsed.push('vectorSearch');
                const vectorSearchResult = await this.performVectorSearch();
                enrichmentResults.vectorSearch = {
                    success: vectorSearchResult.success,
                    similarRecords: vectorSearchResult.data,
                    count: vectorSearchResult.data?.length,
                };
                if (vectorSearchResult.success && vectorSearchResult.data) {
                    enrichedRecord.similarRecords = vectorSearchResult.data;
                }
            }
            // Step 3: Database Lookup
            if (sources.databaseLookup) {
                console.log('[DataEnrichment] Step 3: Database lookup');
                sourcesUsed.push('databaseLookup');
                const dbResult = await this.performDatabaseLookup();
                enrichmentResults.databaseLookup = {
                    success: dbResult.success,
                    data: dbResult.data,
                    rowsAffected: dbResult.rowsAffected,
                };
                if (dbResult.success && dbResult.data) {
                    // Merge database results based on output format
                    this.mergeData(enrichedRecord, dbResult.data, this.params.outputFormat ?? 'merged');
                }
            }
            // Step 4: AI Analysis (always enabled by default)
            if (sources.aiAnalysis !== false) {
                console.log('[DataEnrichment] Step 4: AI analysis');
                sourcesUsed.push('aiAnalysis');
                const aiResult = await this.performAIAnalysis(enrichedRecord, enrichmentResults);
                enrichmentResults.aiAnalysis = {
                    success: aiResult.success,
                    insights: aiResult.insights,
                    confidence: aiResult.confidence,
                };
                if (aiResult.success && aiResult.insights) {
                    enrichedRecord.aiInsights = aiResult.insights;
                    enrichedRecord.aiConfidence = aiResult.confidence;
                }
            }
            // Step 5: Calculate metadata
            const processingTime = Date.now() - startTime;
            const fieldsAdded = Object.keys(enrichedRecord).length - Object.keys(this.params.record).length;
            const dataQualityScore = this.calculateDataQualityScore(enrichedRecord, enrichmentResults);
            console.log(`[DataEnrichment] Enrichment completed in ${processingTime}ms`);
            console.log(`[DataEnrichment] Added ${fieldsAdded} fields`);
            console.log(`[DataEnrichment] Data quality score: ${dataQualityScore.toFixed(2)}`);
            return {
                success: true,
                error: '',
                enrichedRecord,
                enrichmentResults,
                metadata: {
                    sourcesUsed,
                    enrichmentTimestamp: new Date(),
                    processingTime,
                    fieldsAdded,
                    dataQualityScore,
                },
            };
        }
        catch (error) {
            const processingTime = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error('[DataEnrichment] Workflow failed:', errorMessage);
            return {
                success: false,
                error: `Data enrichment failed: ${errorMessage}`,
                enrichedRecord,
                enrichmentResults,
                metadata: {
                    sourcesUsed,
                    enrichmentTimestamp: new Date(),
                    processingTime,
                    fieldsAdded: 0,
                    dataQualityScore: 0,
                },
            };
        }
    }
    /**
     * Perform web search enrichment
     */
    async performWebSearch() {
        try {
            const config = this.params.webSearchConfig;
            if (!config) {
                return { success: false };
            }
            // Generate search query from record
            const searchQuery = config.searchQuery ||
                this.generateSearchQueryFromRecord(this.params.record);
            console.log(`[DataEnrichment] Web search query: ${searchQuery}`);
            // Use HTTP bubble to call search API
            const searchUrl = this.buildSearchUrl(config.searchEngine ?? 'google', searchQuery, config.maxResults ?? 10);
            const httpBubble = new HttpBubble({
                url: searchUrl,
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                },
                timeout: 15000,
                credentials: this.params.credentials,
            }, this.context);
            const result = await httpBubble.action();
            if (result.success && result.data.json) {
                // Extract search results based on search engine
                const results = this.extractSearchResults(result.data.json ?? '', config.searchEngine ?? 'google');
                return { success: true, data: results };
            }
            return { success: false };
        }
        catch (error) {
            console.error('[DataEnrichment] Web search failed:', error);
            return { success: false };
        }
    }
    /**
     * Perform vector similarity search
     */
    async performVectorSearch() {
        try {
            const config = this.params.vectorSearchConfig;
            if (!config) {
                return { success: false };
            }
            console.log(`[DataEnrichment] Vector search: topK=${config.topK}`);
            // Create embedding from record
            const embeddingText = JSON.stringify(this.params.record);
            // Use HTTP bubble to call vector search API
            const httpBubble = new HttpBubble({
                url: config.vectorEndpoint,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: {
                    query: embeddingText,
                    topK: config.topK,
                    threshold: config.threshold,
                },
                timeout: 15000,
                credentials: this.params.credentials,
            }, this.context);
            const result = await httpBubble.action();
            if (result.success && result.data.json) {
                return { success: true, data: result.data.json };
            }
            return { success: false };
        }
        catch (error) {
            console.error('[DataEnrichment] Vector search failed:', error);
            return { success: false };
        }
    }
    /**
     * Perform database lookup
     */
    async performDatabaseLookup() {
        try {
            const config = this.params.databaseConfig;
            if (!config) {
                return { success: false };
            }
            console.log('[DataEnrichment] Database lookup');
            // Use PostgreSQL bubble for database lookup
            const { PostgreSQLBubble } = await import('../service-bubble/postgresql.js');
            const postgresqlBubble = new PostgreSQLBubble({
                query: config.query,
                credentials: this.params.credentials,
            }, this.context);
            const result = await postgresqlBubble.action();
            if (result.success && result.data?.rows) {
                return {
                    success: true,
                    data: result.data.rows[0], // Return first row
                    rowsAffected: result.data.rows.length,
                };
            }
            return { success: false };
        }
        catch (error) {
            console.error('[DataEnrichment] Database lookup failed:', error);
            return { success: false };
        }
    }
    /**
     * Perform AI analysis and synthesis
     */
    async performAIAnalysis(record, enrichmentResults) {
        try {
            const config = this.params.aiConfig;
            const analysisPrompt = config?.analysisPrompt ||
                this.buildDefaultAnalysisPrompt(record, enrichmentResults);
            const aiAgentBubble = new AIAgentBubble({
                message: analysisPrompt,
                systemPrompt: `You are an expert data analyst specializing in data enrichment and synthesis. Your role is to:

1. Analyze the original record alongside all enrichment data
2. Identify key insights, patterns, and relationships
3. Assess the quality and relevance of enriched information
4. Provide actionable intelligence derived from the data
5. Highlight any data quality concerns or anomalies

Focus on delivering concise, high-value insights that help understand the enriched record better. Be specific and evidence-based in your analysis.

Provide your response in the following format:
INSIGHTS: [Your key insights here]
CONFIDENCE: [0.0-1.0 score based on data quality and completeness]
NOTES: [Any concerns or additional context]`,
                model: {
                    model: config?.model || 'google/gemini-2.5-flash',
                    temperature: config?.temperature || 0.3,
                    maxTokens: config?.maxTokens || 50000,
                },
                credentials: this.params.credentials,
            }, this.context);
            const result = await aiAgentBubble.action();
            if (result.success && result.data?.response) {
                const response = result.data.response;
                // Parse confidence from response
                const confidenceMatch = response.match(/CONFIDENCE:\s*([0-9.]+)/);
                const confidence = confidenceMatch
                    ? parseFloat(confidenceMatch[1])
                    : 0.8;
                // Extract insights
                const insightsMatch = response.match(/INSIGHTS:\s*([\s\S]*?)(?=CONFIDENCE:|NOTES:|$)/);
                const insights = insightsMatch
                    ? insightsMatch[1].trim()
                    : response;
                return {
                    success: true,
                    insights,
                    confidence,
                };
            }
            return { success: false };
        }
        catch (error) {
            console.error('[DataEnrichment] AI analysis failed:', error);
            return { success: false };
        }
    }
    /**
     * Generate search query from record
     */
    generateSearchQueryFromRecord(record) {
        const keys = Object.keys(record);
        const searchableKeys = keys.filter(k => k.toLowerCase().includes('name') ||
            k.toLowerCase().includes('company') ||
            k.toLowerCase().includes('title') ||
            k.toLowerCase().includes('organization'));
        if (searchableKeys.length > 0) {
            return searchableKeys
                .map(k => `${k}:${record[k]}`)
                .join(' ');
        }
        // Fallback: use first few values
        return keys.slice(0, 3).map(k => String(record[k])).join(' ');
    }
    /**
     * Build search URL for different search engines
     */
    buildSearchUrl(searchEngine, query, maxResults) {
        const encodedQuery = encodeURIComponent(query);
        switch (searchEngine) {
            case 'google':
                return `https://www.googleapis.com/customsearch/v1?key=YOUR_API_KEY&cx=YOUR_CX&q=${encodedQuery}&num=${maxResults}`;
            case 'bing':
                return `https://api.bing.microsoft.com/v7.0/search?q=${encodedQuery}&count=${maxResults}`;
            case 'duckduckgo':
                // DuckDuckGo doesn't have an official API, this is a placeholder
                return `https://api.duckduckgo.com/?q=${encodedQuery}&format=json`;
            default:
                throw new Error(`Unsupported search engine: ${searchEngine}`);
        }
    }
    /**
     * Extract search results from search engine response
     */
    extractSearchResults(response, searchEngine) {
        if (typeof response !== 'object' || response === null) {
            return [];
        }
        const resp = response;
        switch (searchEngine) {
            case 'google':
                return resp.items || [];
            case 'bing':
                return resp.webPages?.value || [];
            case 'duckduckgo':
                return resp.RelatedTopics || [];
            default:
                return [];
        }
    }
    /**
     * Build default AI analysis prompt
     */
    buildDefaultAnalysisPrompt(record, enrichmentResults) {
        let prompt = `Please analyze the following data record and enrichment results:\n\n`;
        prompt += `**Original Record:**\n${JSON.stringify(record, null, 2)}\n\n`;
        if (enrichmentResults?.webSearch?.results) {
            prompt += `**Web Search Results:**\n${JSON.stringify(enrichmentResults.webSearch.results, null, 2)}\n\n`;
        }
        if (enrichmentResults?.vectorSearch?.similarRecords) {
            prompt += `**Similar Records:**\n${JSON.stringify(enrichmentResults.vectorSearch.similarRecords, null, 2)}\n\n`;
        }
        if (enrichmentResults?.databaseLookup?.data) {
            prompt += `**Database Lookup Results:**\n${JSON.stringify(enrichmentResults.databaseLookup.data, null, 2)}\n\n`;
        }
        prompt += `Provide comprehensive insights about this enriched record.`;
        return prompt;
    }
    /**
     * Merge data based on output format
     */
    mergeData(target, source, format) {
        if (typeof source !== 'object' || source === null) {
            return;
        }
        const src = source;
        switch (format) {
            case 'merged':
                // Merge fields, source overwrites target on conflicts
                Object.assign(target, src);
                break;
            case 'append':
                // Add all fields with prefix
                Object.keys(src).forEach(key => {
                    target[`enriched_${key}`] = src[key];
                });
                break;
            case 'replace':
                // Replace entire record
                Object.keys(target).forEach(key => delete target[key]);
                Object.assign(target, src);
                break;
        }
    }
    /**
     * Calculate data quality score
     */
    calculateDataQualityScore(record, enrichmentResults) {
        let score = 0.5; // Base score
        const maxScore = 1.0;
        // Check for completeness
        const fields = Object.keys(record);
        const nonEmptyFields = fields.filter(k => {
            const v = record[k];
            return v !== null && v !== undefined && v !== '';
        });
        score += (nonEmptyFields.length / Math.max(fields.length, 1)) * 0.2;
        // Check enrichment success
        if (enrichmentResults) {
            const sources = Object.keys(enrichmentResults);
            const successfulSources = sources.filter(s => enrichmentResults[s]?.success);
            score += (successfulSources.length / Math.max(sources.length, 1)) * 0.3;
        }
        return Math.min(score, maxScore);
    }
}
//# sourceMappingURL=data-enrichment.workflow.js.map