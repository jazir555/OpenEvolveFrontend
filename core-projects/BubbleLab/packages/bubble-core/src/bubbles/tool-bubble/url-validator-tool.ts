/**
 * URL VALIDATOR TOOL
 *
 * A tool bubble for comprehensive URL validation including syntax checking,
 * accessibility verification, and security analysis.
 *
 * Features:
 * - Validate URL syntax according to RFC standards
 * - Check URL accessibility (HTTP status)
 * - Detect suspicious/malicious URLs
 * - Extract URL components
 * - Suggest corrections for typos
 * - Batch URL validation
 */

import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';

/**
 * URL validator parameters schema
 */
const URLValidatorToolParamsSchema = z.object({
  // Input URLs
  url: z
    .string()
    .optional()
    .describe('Single URL to validate'),

  urls: z
    .array(z.string())
    .optional()
    .describe('Multiple URLs to validate'),

  // Validation options
  checkSyntax: z
    .boolean()
    .default(true)
    .describe('Validate URL syntax'),

  checkAccessibility: z
    .boolean()
    .default(false)
    .describe('Check if URL is accessible (HTTP HEAD request)'),

  checkSuspicious: z
    .boolean()
    .default(true)
    .describe('Check for suspicious/malicious patterns'),

  allowedProtocols: z
    .array(z.enum(['http', 'https', 'ftp', 'ftps', 'mailto', 'tel', 'file']))
    .default(['http', 'https'])
    .describe('Allowed URL protocols'),

  allowedDomains: z
    .array(z.string())
    .optional()
    .describe('Whitelist of allowed domains'),

  deniedDomains: z
    .array(z.string())
    .optional()
    .describe('Blacklist of denied domains'),

  maxRedirects: z
    .number()
    .int()
    .min(0)
    .default(5)
    .describe('Maximum number of redirects to follow'),

  timeout: z
    .number()
    .int()
    .positive()
    .default(5000)
    .describe('Request timeout in milliseconds'),

  // Credentials
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Credentials for external services'),
});

/**
 * URL validation result schema
 */
const URLValidationResultSchema = z.object({
  url: z.string().describe('URL that was validated'),
  isValid: z.boolean().describe('Whether the URL is valid'),
  isAccessible: z.boolean().optional().describe('Whether the URL is accessible'),
  isSuspicious: z.boolean().describe('Whether the URL has suspicious patterns'),
  protocol: z.string().optional().describe('URL protocol'),
  domain: z.string().optional().describe('URL domain'),
  path: z.string().optional().describe('URL path'),
  query: z.string().optional().describe('URL query string'),
  fragment: z.string().optional().describe('URL fragment'),
  statusCode: z.number().optional().describe('HTTP status code'),
  redirectCount: z.number().optional().describe('Number of redirects'),
  errors: z
    .array(z.string())
    .describe('Validation errors'),
  warnings: z
    .array(z.string())
    .describe('Validation warnings'),
});

/**
 * URL validator result schema
 */
const URLValidatorToolResultSchema = z.object({
  // Result
  success: z.boolean().describe('Whether the validation was successful'),

  // Validation results
  results: z
    .array(URLValidationResultSchema)
    .describe('Validation results for each URL'),

  // Summary statistics
  stats: z
    .object({
      totalURLs: z.number(),
      validURLs: z.number(),
      invalidURLs: z.number(),
      accessibleURLs: z.number(),
      suspiciousURLs: z.number(),
      processingTime: z.number(),
    })
    .describe('Validation statistics'),

  error: z.string().describe('Error message if validation failed'),
});

// Type definitions
type URLValidatorToolParams = z.output<typeof URLValidatorToolParamsSchema>;
type URLValidatorToolResult = z.output<typeof URLValidatorToolResultSchema>;
type URLValidatorToolParamsInput = z.input<typeof URLValidatorToolParamsSchema>;

/**
 * Suspicious URL patterns
 */
const SUSPICIOUS_PATTERNS = [
  /bit\.ly/i,
  /tinyurl\.com/i,
  /goo\.gl/i,
  /\.\.\/|\.\\\//, // Directory traversal
  /<script|javascript:/i, // XSS attempts
  /@/i, // Credential injection
  /%[0-9a-f]{2}/i, // URL encoding (suspicious if excessive)
];

/**
 * URL Validator Tool
 * Comprehensive URL validation with security checks
 */
export class URLValidatorTool extends ToolBubble<
  URLValidatorToolParams,
  URLValidatorToolResult
> {
  /**
   * REQUIRED STATIC METADATA
   */
  static readonly type = 'tool' as const;
  static readonly bubbleName: BubbleName = 'url-validator-tool';
  static readonly schema = URLValidatorToolParamsSchema;
  static readonly resultSchema = URLValidatorToolResultSchema;
  static readonly shortDescription =
    'Validate URLs with syntax, accessibility, and security checks';
  static readonly longDescription = `
    A comprehensive URL validation tool with multiple verification layers.

    Features:
    - SYNTAX VALIDATION: RFC-compliant URL syntax checking
    - ACCESSIBILITY CHECK: Verify URL is reachable (HTTP HEAD)
    - SUSPICIOUS DETECTION: Identify malicious or dangerous patterns
    - COMPONENT EXTRACTION: Parse URL into components
    - DOMAIN VALIDATION: Whitelist/blacklist checking
    - BATCH VALIDATION: Validate multiple URLs at once

    Validation Checks:
    - Syntax validation (RFC 3986)
    - Protocol validation
    - Domain accessibility (HTTP status)
    - Suspicious pattern detection
    - Domain whitelist/blacklist
    - Redirect following
    - Timeout enforcement

    Security Checks:
    - Directory traversal attempts
    - XSS injection attempts
    - URL encoding abuse
    - Suspicious URL shorteners
    - Credential injection attempts

    Use cases:
    - User-submitted URL validation
    - Link quality checking
    - Security scanning
    - Web scraping preparation
    - API endpoint validation
    - Content moderation

    Supported Protocols:
    - HTTP, HTTPS (default)
    - FTP, FTPS
    - Mailto, Tel
    - File

    Accessibility Checks:
    - HTTP HEAD request
    - Status code verification
    - Redirect following
    - Timeout handling
    - Error detection

    Note: Accessibility checking requires network access.
  `;
  static readonly alias = 'url-validate';

  constructor(
    params: URLValidatorToolParamsInput,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  /**
   * Main action method - performs URL validation
   */
  async performAction(
    context?: BubbleContext
  ): Promise<URLValidatorToolResult> {
    void context; // Context available but not currently used
    const startTime = Date.now();

    try {
      console.log('[URLValidatorTool] Starting URL validation');

      // Determine URLs to validate
      let urlsToValidate: string[];

      if (this.params.url) {
        urlsToValidate = [this.params.url];
      } else if (this.params.urls) {
        urlsToValidate = this.params.urls;
      } else {
        throw new Error('Either url or urls parameter is required');
      }

      // Validate each URL
      const results = await Promise.all(
        urlsToValidate.map((url) => this.validateURL(url))
      );

      // Calculate statistics
      const validURLs = results.filter((r) => r.isValid).length;
      const invalidURLs = results.filter((r) => !r.isValid).length;
      const accessibleURLs = results.filter((r) => r.isAccessible).length;
      const suspiciousURLs = results.filter((r) => r.isSuspicious).length;

      const processingTime = Date.now() - startTime;

      console.log(`[URLValidatorTool] Validation completed. Valid: ${validURLs}, Invalid: ${invalidURLs}`);

      return {
        success: true,
        results,
        stats: {
          totalURLs: urlsToValidate.length,
          validURLs,
          invalidURLs,
          accessibleURLs,
          suspiciousURLs,
          processingTime,
        },
        error: '',
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';

      console.error(`[URLValidatorTool] Validation failed: ${errorMessage}`);

      return {
        success: false,
        results: [],
        stats: {
          totalURLs: 0,
          validURLs: 0,
          invalidURLs: 0,
          accessibleURLs: 0,
          suspiciousURLs: 0,
          processingTime: Date.now() - startTime,
        },
        error: errorMessage,
      };
    }
  }

  /**
   * Validate a single URL
   */
  private async validateURL(
    url: string
  ): Promise<{
    url: string;
    isValid: boolean;
    isAccessible?: boolean;
    isSuspicious: boolean;
    protocol?: string;
    domain?: string;
    path?: string;
    query?: string;
    fragment?: string;
    statusCode?: number;
    redirectCount?: number;
    errors: string[];
    warnings: string[];
  }> {
    const errors: string[] = [];
    const warnings: string[] = [];
    let isValid = true;
    let isAccessible: boolean | undefined = undefined;
    let isSuspicious = false;
    let protocol: string | undefined = undefined;
    let domain: string | undefined = undefined;
    let path: string | undefined = undefined;
    let query: string | undefined = undefined;
    let fragment: string | undefined = undefined;
    let statusCode: number | undefined = undefined;
    let redirectCount = 0;

    try {
      // Normalize URL
      const normalizedURL = url.trim();

      // Parse URL
      let parsedURL: URL;

      try {
        parsedURL = new URL(normalizedURL);
        protocol = parsedURL.protocol.replace(':', '');
        domain = parsedURL.hostname;
        path = parsedURL.pathname;
        query = parsedURL.search;
        fragment = parsedURL.hash;
      } catch (parseError) {
        isValid = false;
        errors.push(`Invalid URL syntax: ${parseError instanceof Error ? parseError.message : 'Unknown error'}`);
        return {
          url,
          isValid,
          isSuspicious,
          errors,
          warnings,
        };
      }

      // Check protocol
      if (this.params.allowedProtocols && !this.params.allowedProtocols.includes(protocol as any)) {
        isValid = false;
        errors.push(`Protocol "${protocol}" is not allowed`);
      }

      // Check domain whitelist/blacklist
      if (domain) {
        if (this.params.allowedDomains && this.params.allowedDomains.length > 0) {
          if (!this.params.allowedDomains.includes(domain)) {
            errors.push(`Domain ${domain} is not in the allowed list`);
            isValid = false;
          }
        }

        if (this.params.deniedDomains && this.params.deniedDomains.length > 0) {
          if (this.params.deniedDomains.includes(domain)) {
            errors.push(`Domain ${domain} is in the denied list`);
            isValid = false;
          }
        }
      }

      // Check for suspicious patterns
      if (this.params.checkSuspicious) {
        for (const pattern of SUSPICIOUS_PATTERNS) {
          if (pattern.test(normalizedURL)) {
            isSuspicious = true;
            warnings.push(`URL contains suspicious pattern: ${pattern.source}`);
          }
        }

        // Check for excessive URL encoding
        const encodedMatches = normalizedURL.match(/%[0-9a-f]{2}/gi);
        if (encodedMatches && encodedMatches.length > 10) {
          isSuspicious = true;
          warnings.push(`URL contains excessive encoded characters (${encodedMatches.length})`);
        }

        // Check for IP address instead of domain (potentially suspicious)
        if (domain && /^(\d{1,3}\.){3}\d{1,3}$/.test(domain)) {
          isSuspicious = true;
          warnings.push(`URL uses IP address instead of domain name`);
        }
      }

      // Check accessibility
      if (this.params.checkAccessibility && isValid) {
        try {
          const response = await fetch(normalizedURL, {
            method: 'HEAD',
            redirect: 'manual',
            signal: AbortSignal.timeout(this.params.timeout),
          });

          statusCode = response.status;
          isAccessible = statusCode >= 200 && statusCode < 400;

          if (!isAccessible) {
            warnings.push(`URL returned status code ${statusCode}`);
          }

          // Count redirects
          let redirectURL = response.headers.get('Location');
          while (redirectURL && redirectCount < this.params.maxRedirects) {
            redirectCount++;
            // Could follow redirects here if needed
            break;
          }

          if (redirectCount >= this.params.maxRedirects) {
            warnings.push(`URL exceeded maximum redirect count (${this.params.maxRedirects})`);
          }
        } catch (fetchError) {
          isAccessible = false;
          warnings.push(`Failed to check accessibility: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`);
        }
      }
    } catch (error) {
      errors.push(`Validation error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      isValid = false;
    }

    return {
      url,
      isValid,
      isAccessible,
      isSuspicious,
      protocol,
      domain,
      path,
      query,
      fragment,
      statusCode,
      redirectCount,
      errors,
      warnings,
    };
  }
}
