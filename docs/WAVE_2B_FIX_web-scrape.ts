/**
 * COMPREHENSIVE VALIDATION FIXES FOR web-scrape-tool.ts
 *
 * This file contains all validation improvements to be applied to
 * web-scrape-tool.ts. Replace the existing schema definitions
 * (around lines 37-72) with these enhanced versions.
 */

import { z } from 'zod';
import { CredentialType } from '@bubblelab/shared-schemas';

// ============================================================================
// URL VALIDATION CONSTANTS
// ============================================================================

/**
 * Blocked patterns for security
 * Prevent scraping of internal networks, localhost, private IPs
 */
const BLOCKED_PATTERNS = [
  'localhost',
  '127.0.0.1',
  '0.0.0.0',
  '::1',
  '[::1]',
  '192.168.',
  '10.',
  '172.16.',
  '172.17.',
  '172.18.',
  '172.19.',
  '172.20.',
  '172.21.',
  '172.22.',
  '172.23.',
  '172.24.',
  '172.25.',
  '172.26.',
  '172.27.',
  '172.28.',
  '172.29.',
  '172.30.',
  '172.31.',
  '169.254.', // Link-local
  'fc00:', // Private IPv6
  'fe80:', // Link-local IPv6
  'fd',    // Private IPv6
];

/**
 * Allowed protocols for web scraping
 */
const ALLOWED_PROTOCOLS = ['http:', 'https:'];

// ============================================================================
// ENHANCED URL VALIDATION
// ============================================================================

/**
 * Comprehensive URL validation with security checks
 * - Validates URL format
 * - Restricts to HTTP/HTTPS only
 * - Blocks internal networks, localhost, private IPs
 * - Enforces length limits
 * - Validates domain format
 */
const EnhancedURLSchema = z.string()
  .max(2048, { message: 'URL cannot exceed 2048 characters' })
  .url({ message: 'Must be a valid URL' })
  .refine(
    (url) => {
      try {
        const parsed = new URL(url);
        return ALLOWED_PROTOCOLS.includes(parsed.protocol);
      } catch {
        return false;
      }
    },
    { message: 'Only HTTP and HTTPS protocols are allowed' }
  )
  .refine(
    (url) => {
      const lowerUrl = url.toLowerCase();
      return !BLOCKED_PATTERNS.some(pattern => lowerUrl.includes(pattern));
    },
    { message: 'URLs containing localhost, private IPs, or internal networks are not allowed' }
  )
  .refine(
    (url) => {
      try {
        const parsed = new URL(url);
        // Validate domain has at least one dot and valid TLD
        const domain = parsed.hostname;
        if (!domain || !domain.includes('.')) {
          return false;
        }
        // Basic TLD validation (at least 2 chars)
        const tld = domain.split('.').pop();
        return !!tld && tld.length >= 2;
      } catch {
        return false;
      }
    },
    { message: 'URL must have a valid domain name with TLD' }
  )
  .refine(
    (url) => {
      // Check for suspicious patterns
      const suspicious = [
        'file:',
        'ftp:',
        'data:',
        'javascript:',
        'vbscript:',
        'about:',
        'mailto:',
        'ssh:',
        'telnet:',
      ];
      return !suspicious.some(pattern => url.toLowerCase().includes(pattern));
    },
    { message: 'URL contains suspicious or disallowed protocol' }
  )
  .describe('HTTP/HTTPS URL to scrape (max 2048 chars, public URLs only)');

// ============================================================================
// CREDENTIALS VALIDATION
// ============================================================================

/**
 * Validates API credentials
 * - Ensures required credentials are present
 * - Validates credential formats
 * - Checks minimum lengths
 */
const CredentialsSchema = z.record(
  z.nativeEnum(CredentialType),
  z.string()
    .min(1, { message: 'Credential value cannot be empty' })
    .max(4096, { message: 'Credential value cannot exceed 4096 characters' })
).refine(
  (credentials) => {
    // If FIRECRAWL_API_KEY is provided, validate its format
    if (credentials.FIRECRAWL_API_KEY) {
      const key = credentials.FIRECRAWL_API_KEY;
      // Basic format validation (typically 20+ chars, alphanumeric with some special chars)
      if (key.length < 20) {
        return false;
      }
      // Check for reasonable character set
      if (!/^[A-Za-z0-9_-]+$/.test(key)) {
        return false;
      }
    }
    return true;
  },
  {
    message: 'FIRECRAWL_API_KEY must be at least 20 characters and contain only alphanumeric characters, hyphens, underscores'
  }
).refine(
  (credentials) => {
    // Ensure at least one credential is provided if credentials object exists
    return Object.keys(credentials).length > 0;
  },
  {
    message: 'At least one credential must be provided'
  }
).optional()
  .describe('API credentials including FIRECRAWL_API_KEY');

// ============================================================================
// PARAMETERS SCHEMA
// ============================================================================

/**
 * Enhanced web scrape tool parameters with comprehensive validation
 */
const WebScrapeToolParamsSchema = z.object({
  // URL with enhanced validation
  url: EnhancedURLSchema,

  // Format: validated enum
  format: z.enum(['markdown', 'html', 'rawHtml', 'cleaned'], {
    requiredError: 'Format is required',
    invalidTypeError: 'Format must be one of: markdown, html, rawHtml, cleaned'
  })
  .default('markdown')
  .describe('Content format to extract'),

  // Only main content: validated boolean
  onlyMainContent: z.boolean()
    .default(true)
    .describe('Extract only main content, filtering out navigation/footers'),

  // Credentials with validation
  credentials: CredentialsSchema
});

// ============================================================================
// RESULT SCHEMA
// ============================================================================

/**
 * Validates API response from Firecrawl
 * - Ensures data structure matches expectations
 * - Validates content size limits
 * - Validates status codes
 * - Validates metadata structure
 */
const FirecrawlResponseSchema = z.object({
  data: z.object({
    markdown: z.string()
      .max(1e8, { message: 'Markdown content cannot exceed 100MB' })
      .optional(),
    html: z.string()
      .max(1e8, { message: 'HTML content cannot exceed 100MB' })
      .optional(),
    rawHtml: z.string()
      .max(1e8, { message: 'Raw HTML content cannot exceed 100MB' })
      .optional(),
    metadata: z.object({
      title: z.string()
        .max(256, { message: 'Title cannot exceed 256 characters' })
        .optional(),
      description: z.string()
        .max(1024, { message: 'Description cannot exceed 1024 characters' })
        .optional(),
      keywords: z.string()
        .max(512, { message: 'Keywords cannot exceed 512 characters' })
        .optional(),
      author: z.string()
        .max(128, { message: 'Author cannot exceed 128 characters' })
        .optional(),
      statusCode: z.number()
        .int({ message: 'Status code must be an integer' })
        .min(100, { message: 'Status code must be at least 100' })
        .max(599, { message: 'Status code cannot exceed 599' })
        .optional(),
      language: z.string()
        .max(32, { message: 'Language cannot exceed 32 characters' })
        .optional(),
    }).optional()
  }).optional(),
  success: z.boolean({ requiredError: 'Success flag is required' }),
  error: z.string()
    .max(10000, { message: 'Error message cannot exceed 10000 characters' })
    .optional()
});

/**
 * Web scrape tool result schema
 * Validates all output fields
 */
const WebScrapeToolResultSchema = z.object({
  content: z.string()
    .max(1e8, { message: 'Content cannot exceed 100MB' })
    .describe('Scraped content in requested format'),

  title: z.string()
    .max(256, { message: 'Title cannot exceed 256 characters' })
    .describe('Page title if available'),

  url: z.string()
    .url({ message: 'Must be a valid URL' })
    .max(2048, { message: 'URL cannot exceed 2048 characters' })
    .describe('The original URL that was scraped'),

  format: z.enum(['markdown', 'html', 'rawHtml', 'cleaned'])
    .describe('Format of the returned content'),

  success: z.boolean({ requiredError: 'Success flag is required' })
    .describe('Whether the scraping was successful'),

  error: z.string()
    .max(10000, { message: 'Error message cannot exceed 10000 characters' })
    .describe('Error message if scraping failed'),

  creditsUsed: z.number()
    .int({ message: 'Credits used must be an integer' })
    .min(0, { message: 'Credits used cannot be negative' })
    .max(1000, { message: 'Credits used cannot exceed 1000' })
    .describe('Number of credits used'),

  metadata: z.object({
    statusCode: z.number()
      .int({ message: 'Status code must be an integer' })
      .min(100, { message: 'Status code must be at least 100' })
      .max(599, { message: 'Status code cannot exceed 599' })
      .optional()
      .describe('HTTP status code'),

    loadTime: z.number()
      .int({ message: 'Load time must be an integer' })
      .min(0, { message: 'Load time cannot be negative' })
      .max(600000, { message: 'Load time cannot exceed 600000ms (10 minutes)' })
      .optional()
      .describe('Page load time in milliseconds')
  }).optional()
    .describe('Additional metadata about the scrape')
});

// ============================================================================
// VALIDATION HELPER FUNCTIONS
// ============================================================================

/**
 * Validates Firecrawl API response
 * Ensures response structure matches expected schema
 */
function validateFirecrawlResponse(response: any): {
  valid: boolean;
  error?: string;
  validated?: any;
} {
  try {
    const validated = FirecrawlResponseSchema.parse(response);
    return { valid: true, validated };
  } catch (error) {
    if (error instanceof z.ZodError) {
      const formattedErrors = error.errors.map((err) => {
        const path = err.path.join('.') || 'root';
        return `${path}: ${err.message}`;
      }).join('; ');

      return {
        valid: false,
        error: `Invalid API response: ${formattedErrors}`
      };
    }
    return {
      valid: false,
      error: 'Unknown validation error'
    };
  }
}

/**
 * Validates web scrape result before returning
 * Ensures all output fields are valid
 */
function validateWebScrapeResult(result: any): {
  valid: boolean;
  error?: string;
  validated?: any;
} {
  try {
    const validated = WebScrapeToolResultSchema.parse(result);
    return { valid: true, validated };
  } catch (error) {
    if (error instanceof z.ZodError) {
      const formattedErrors = error.errors.map((err) => {
        const path = err.path.join('.') || 'root';
        return `${path}: ${err.message}`;
      }).join('; ');

      return {
        valid: false,
        error: `Invalid result: ${formattedErrors}`
      };
    }
    return {
      valid: false,
      error: 'Unknown validation error'
    };
  }
}

/**
 * Sanitizes scraped content
 * Removes or escapes potentially dangerous content
 */
function sanitizeContent(content: string, format: string): string {
  if (!content) return '';

  // Trim whitespace
  let sanitized = content.trim();

  // For HTML/rawHtml, perform additional sanitization
  if (format === 'html' || format === 'rawHtml') {
    // Remove script tags (basic XSS prevention)
    sanitized = sanitized.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');

    // Remove iframe tags
    sanitized = sanitized.replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '');

    // Remove object/embed tags
    sanitized = sanitized.replace(/<(object|embed)\b[^<]*(?:(?!<\/\1>)<[^<]*)*<\/\1>/gi, '');
  }

  // Limit content size
  const maxSize = 5 * 1024 * 1024; // 5MB
  if (sanitized.length > maxSize) {
    sanitized = sanitized.substring(0, maxSize) + '\n\n...[Content truncated due to size limit]...';
  }

  return sanitized;
}

/**
 * Validates content size before processing
 * Returns true if content size is acceptable
 */
function validateContentSize(content: string, maxSize: number = 1e8): boolean {
  if (!content) return true;
  return content.length <= maxSize;
}

// ============================================================================
// USAGE IN CLASS
// ============================================================================

/**
 * In the WebScrapeTool class, update the performAction method:
 *
 * async performAction(): Promise<WebScrapeToolResult> {
 *   const { url, format, credentials } = this.params;
 *   const startTime = Date.now();
 *
 *   try {
 *     console.debug('[WebScrapeTool] Scraping URL:', url, 'with format:', format);
 *
 *     const firecrawl = new FirecrawlBubble(
 *       {
 *         operation: 'scrape' as const,
 *         credentials,
 *         url,
 *         formats: [format],
 *         waitFor: 2000,
 *         maxAge: 172800000,
 *         parsers: ['pdf'],
 *       },
 *       this.context,
 *       'web_scrape_tool_firecrawl'
 *     );
 *
 *     // Execute scrape
 *     const rawResponse = await firecrawl.action();
 *
 *     // VALIDATE RESPONSE STRUCTURE
 *     const responseValidation = validateFirecrawlResponse(rawResponse);
 *     if (!responseValidation.valid) {
 *       console.error('[WebScrapeTool] Invalid API response:', responseValidation.error);
 *       return {
 *         content: '',
 *         title: '',
 *         url,
 *         format,
 *         success: false,
 *         error: responseValidation.error || 'Invalid API response',
 *         creditsUsed: 0,
 *         metadata: { loadTime: Date.now() - startTime }
 *       };
 *     }
 *
 *     const response = responseValidation.validated!;
 *
 *     // Extract content based on format
 *     let content: string;
 *     let title = '';
 *
 *     if (format === 'markdown' && response.data?.markdown) {
 *       content = response.data.markdown;
 *     } else if (format === 'html' && response.data?.html) {
 *       content = response.data.html;
 *     } else if (format === 'rawHtml' && response.data?.rawHtml) {
 *       content = response.data.rawHtml;
 *     } else if (format === 'cleaned' && response.data?.markdown) {
 *       content = response.data.markdown;
 *     } else {
 *       throw new Error(`No content available in ${format} format`);
 *     }
 *
 *     // VALIDATE CONTENT SIZE
 *     if (!validateContentSize(content)) {
 *       throw new Error('Content size exceeds maximum allowed size');
 *     }
 *
 *     // SANITIZE CONTENT
 *     content = sanitizeContent(content, format);
 *
 *     // Summarize the scraped content for better consumption
 *     if (content && content.length > 5000000) {
 *       try {
 *         const summarizeAgent = new AIAgentBubble(
 *           {
 *             message: `Summarize the scraped content... Content: ${content}`,
 *             model: {
 *               model: 'google/gemini-2.5-flash-lite',
 *               maxTokens: 80000,
 *             },
 *             name: 'Scrape Content Summarizer Agent',
 *             credentials: this.params.credentials,
 *           },
 *           this.context
 *         );
 *
 *         const result = await summarizeAgent.action();
 *         if (result.data?.response) {
 *           console.log('[WebScrapeTool] Summarized scraped content for:', url);
 *           content = sanitizeContent(result.data.response, format);
 *         }
 *       } catch (error) {
 *         console.error('[WebScrapeTool] Error summarizing content:', url, error);
 *         // Keep original content if summarization fails
 *       }
 *     }
 *
 *     // Extract title from metadata
 *     if (response.data?.metadata?.title) {
 *       title = response.data.metadata.title;
 *     }
 *
 *     const loadTime = Date.now() - startTime;
 *
 *     // VALIDATE RESULT BEFORE RETURNING
 *     const rawResult = {
 *       content: content.trim(),
 *       title,
 *       url,
 *       creditsUsed: 1,
 *       format,
 *       success: true,
 *       error: '',
 *       metadata: {
 *         statusCode: response.data?.metadata?.statusCode,
 *         loadTime,
 *       },
 *     };
 *
 *     const resultValidation = validateWebScrapeResult(rawResult);
 *     if (!resultValidation.valid) {
 *       console.error('[WebScrapeTool] Invalid result:', resultValidation.error);
 *       throw new Error(resultValidation.error);
 *     }
 *
 *     return resultValidation.validated!;
 *   } catch (error) {
 *     console.error('[WebScrapeTool] Scrape error:', error);
 *
 *     const errorMessage = error instanceof Error ? error.message : 'Unknown error';
 *
 *     const rawResult = {
 *       content: '',
 *       title: '',
 *       url,
 *       format,
 *       success: false,
 *       error: errorMessage,
 *       creditsUsed: 0,
 *       metadata: {
 *         loadTime: Date.now() - startTime,
 *       },
 *     };
 *
 *     const resultValidation = validateWebScrapeResult(rawResult);
 *     return resultValidation.validated || rawResult;
 *   }
 * }
 */

export {
  EnhancedURLSchema,
  CredentialsSchema,
  WebScrapeToolParamsSchema,
  WebScrapeToolResultSchema,
  FirecrawlResponseSchema,
  validateFirecrawlResponse,
  validateWebScrapeResult,
  sanitizeContent,
  validateContentSize
};
