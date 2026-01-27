import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { FirecrawlBubble } from '../service-bubble/firecrawl.js';
import { CredentialType } from '@bubblelab/shared-schemas';
// Parameters schema for web extraction
const WebExtractToolParamsSchema = z.object({
    url: z
        .string()
        .url('Must be a valid URL')
        .describe('The URL to extract structured data from'),
    prompt: z
        .string()
        .min(1, 'Extraction prompt is required')
        .describe('Detailed prompt describing what data to extract from the web page'),
    schema: z
        .string()
        .min(1, 'JSON schema is required')
        .describe('JSON schema string defining the structure of the data to extract'),
    timeout: z
        .number()
        .min(1000)
        .max(60000)
        .default(30000)
        .optional()
        .describe('Timeout in milliseconds for the extraction (default: 30000)'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Required credentials including FIRECRAWL_API_KEY'),
});
// Result schema for extraction operations
const WebExtractToolResultSchema = z.object({
    url: z.string().url().describe('The original URL that was processed'),
    success: z.boolean().describe('Whether the extraction was successful'),
    error: z.string().describe('Error message if extraction failed'),
    extractedData: z
        .any()
        .describe('The extracted structured data matching the provided schema'),
    metadata: z
        .object({
        extractionTime: z.number().optional(),
        pageTitle: z.string().optional(),
        statusCode: z.number().optional(),
    })
        .optional()
        .describe('Additional metadata about the extraction'),
});
export class WebExtractTool extends ToolBubble {
    // Required static metadata
    static bubbleName = 'web-extract-tool';
    static schema = WebExtractToolParamsSchema;
    static resultSchema = WebExtractToolResultSchema;
    static shortDescription = 'Extracts structured data from web pages using Firecrawl AI-powered extraction with custom prompts and schemas';
    static longDescription = `
    A powerful web data extraction tool that uses Firecrawl's AI-powered extraction API to extract structured data from web pages.
    
    🎯 EXTRACT Features:
    - AI-powered structured data extraction using natural language prompts
    - Custom JSON schema validation for extracted data
    - Handles dynamic content and JavaScript-rendered pages
    - Precise extraction of specific elements like images, prices, descriptions
    - Works with complex e-commerce sites and product pages
    - Requires FIRECRAWL_API_KEY credential
    
    Use Cases:
    - Extract product information (names, prices, images) from e-commerce sites
    - Gather structured data from listings and catalogs
    - Extract contact information and business details
    - Parse article metadata and content structure
    - Collect specific data points from forms and tables
    - Extract image URLs, especially for product galleries
    
    How it works:
    1. Provide a URL and a natural language prompt describing what to extract
    2. Define a JSON schema for the expected data structure
    3. Firecrawl's AI analyzes the page and extracts matching data
    4. Returns structured data validated against your schema
    
    Example use case:
    - URL: Uniqlo product page
    - Prompt: "Extract the product name, price, and all available product image URLs"
    - Schema: {"name": "string", "price": "number", "images": ["string"]}
    - Result: Structured JSON with the exact data you need
  `;
    static alias = 'extract';
    static type = 'tool';
    constructor(params = {
        url: '',
        prompt: '',
        schema: '{}',
    }, context) {
        super(params, context);
    }
    async performAction(context) {
        void context; // Context available but not currently used
        const { url, prompt, schema, timeout, credentials } = this.params;
        const startTime = Date.now();
        try {
            console.log('[WebExtractTool] Extracting data from URL:', url);
            console.log('[WebExtractTool] Using prompt:', prompt.substring(0, 100) + '...');
            console.log('[WebExtractTool] Expected schema:', schema.substring(0, 200) + '...');
            // Validate and parse the JSON schema
            let parsedSchema;
            try {
                parsedSchema = JSON.parse(schema);
            }
            catch (parseError) {
                throw new Error(`Invalid JSON schema provided: ${parseError instanceof Error ? parseError.message : 'Unknown parsing error'}`);
            }
            // Configure extraction options
            const extractOptions = {
                urls: [url],
                prompt,
                schema: parsedSchema,
                timeout: timeout || 30000, // Timeout in milliseconds (default 30s)
            };
            // Initialize Firecrawl bubble
            const firecrawlParams = {
                operation: 'extract',
                credentials,
                ...extractOptions,
            };
            const firecrawl = new FirecrawlBubble(firecrawlParams, this.context, 'web_extract_tool_firecrawl');
            // Execute extraction
            const response = await firecrawl.action();
            // Handle the response
            if (!response.data || typeof response.data !== 'object') {
                throw new Error('Invalid response from Firecrawl extract API');
            }
            // Check if extraction was successful
            if (response.success === false || response.data.status === 'failed') {
                throw new Error(response.error || 'Extraction failed');
            }
            // Extract the data from the response
            let extractedData = {};
            if (response.data.data) {
                extractedData = response.data.data;
            }
            else {
                throw new Error('No data returned from extraction');
            }
            const extractionTime = Date.now() - startTime;
            // Extract metadata if available
            const metadata = {
                extractionTime,
            };
            if (response.data.sources) {
                metadata.sources = response.data.sources;
            }
            console.log('[WebExtractTool] Successfully extracted data from:', url);
            console.log('[WebExtractTool] Extraction time:', extractionTime, 'ms');
            return {
                url,
                success: true,
                error: '',
                extractedData,
                metadata,
            };
        }
        catch (error) {
            console.error('[WebExtractTool] Extraction error:', error);
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            return {
                url,
                success: false,
                error: errorMessage,
                extractedData: {},
                metadata: {
                    extractionTime: Date.now() - startTime,
                },
            };
        }
    }
}
//# sourceMappingURL=web-extract-tool.js.map