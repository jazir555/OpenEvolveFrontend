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
declare const URLValidatorToolParamsSchema: z.ZodObject<{
    url: z.ZodOptional<z.ZodString>;
    urls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    checkSyntax: z.ZodDefault<z.ZodBoolean>;
    checkAccessibility: z.ZodDefault<z.ZodBoolean>;
    checkSuspicious: z.ZodDefault<z.ZodBoolean>;
    allowedProtocols: z.ZodDefault<z.ZodArray<z.ZodEnum<["http", "https", "ftp", "ftps", "mailto", "tel", "file"]>, "many">>;
    allowedDomains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    deniedDomains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    maxRedirects: z.ZodDefault<z.ZodNumber>;
    timeout: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    maxRedirects: number;
    checkSyntax: boolean;
    checkAccessibility: boolean;
    checkSuspicious: boolean;
    allowedProtocols: ("http" | "file" | "https" | "ftp" | "ftps" | "mailto" | "tel")[];
    url?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    urls?: string[] | undefined;
    allowedDomains?: string[] | undefined;
    deniedDomains?: string[] | undefined;
}, {
    timeout?: number | undefined;
    url?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    urls?: string[] | undefined;
    maxRedirects?: number | undefined;
    checkSyntax?: boolean | undefined;
    allowedDomains?: string[] | undefined;
    deniedDomains?: string[] | undefined;
    checkAccessibility?: boolean | undefined;
    checkSuspicious?: boolean | undefined;
    allowedProtocols?: ("http" | "file" | "https" | "ftp" | "ftps" | "mailto" | "tel")[] | undefined;
}>;
/**
 * URL validator result schema
 */
declare const URLValidatorToolResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    results: z.ZodArray<z.ZodObject<{
        url: z.ZodString;
        isValid: z.ZodBoolean;
        isAccessible: z.ZodOptional<z.ZodBoolean>;
        isSuspicious: z.ZodBoolean;
        protocol: z.ZodOptional<z.ZodString>;
        domain: z.ZodOptional<z.ZodString>;
        path: z.ZodOptional<z.ZodString>;
        query: z.ZodOptional<z.ZodString>;
        fragment: z.ZodOptional<z.ZodString>;
        statusCode: z.ZodOptional<z.ZodNumber>;
        redirectCount: z.ZodOptional<z.ZodNumber>;
        errors: z.ZodArray<z.ZodString, "many">;
        warnings: z.ZodArray<z.ZodString, "many">;
    }, "strip", z.ZodTypeAny, {
        url: string;
        errors: string[];
        isValid: boolean;
        warnings: string[];
        isSuspicious: boolean;
        path?: string | undefined;
        query?: string | undefined;
        domain?: string | undefined;
        statusCode?: number | undefined;
        protocol?: string | undefined;
        isAccessible?: boolean | undefined;
        fragment?: string | undefined;
        redirectCount?: number | undefined;
    }, {
        url: string;
        errors: string[];
        isValid: boolean;
        warnings: string[];
        isSuspicious: boolean;
        path?: string | undefined;
        query?: string | undefined;
        domain?: string | undefined;
        statusCode?: number | undefined;
        protocol?: string | undefined;
        isAccessible?: boolean | undefined;
        fragment?: string | undefined;
        redirectCount?: number | undefined;
    }>, "many">;
    stats: z.ZodObject<{
        totalURLs: z.ZodNumber;
        validURLs: z.ZodNumber;
        invalidURLs: z.ZodNumber;
        accessibleURLs: z.ZodNumber;
        suspiciousURLs: z.ZodNumber;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        totalURLs: number;
        validURLs: number;
        invalidURLs: number;
        accessibleURLs: number;
        suspiciousURLs: number;
    }, {
        processingTime: number;
        totalURLs: number;
        validURLs: number;
        invalidURLs: number;
        accessibleURLs: number;
        suspiciousURLs: number;
    }>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        totalURLs: number;
        validURLs: number;
        invalidURLs: number;
        accessibleURLs: number;
        suspiciousURLs: number;
    };
    results: {
        url: string;
        errors: string[];
        isValid: boolean;
        warnings: string[];
        isSuspicious: boolean;
        path?: string | undefined;
        query?: string | undefined;
        domain?: string | undefined;
        statusCode?: number | undefined;
        protocol?: string | undefined;
        isAccessible?: boolean | undefined;
        fragment?: string | undefined;
        redirectCount?: number | undefined;
    }[];
}, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        totalURLs: number;
        validURLs: number;
        invalidURLs: number;
        accessibleURLs: number;
        suspiciousURLs: number;
    };
    results: {
        url: string;
        errors: string[];
        isValid: boolean;
        warnings: string[];
        isSuspicious: boolean;
        path?: string | undefined;
        query?: string | undefined;
        domain?: string | undefined;
        statusCode?: number | undefined;
        protocol?: string | undefined;
        isAccessible?: boolean | undefined;
        fragment?: string | undefined;
        redirectCount?: number | undefined;
    }[];
}>;
type URLValidatorToolParams = z.output<typeof URLValidatorToolParamsSchema>;
type URLValidatorToolResult = z.output<typeof URLValidatorToolResultSchema>;
type URLValidatorToolParamsInput = z.input<typeof URLValidatorToolParamsSchema>;
/**
 * URL Validator Tool
 * Comprehensive URL validation with security checks
 */
export declare class URLValidatorTool extends ToolBubble<URLValidatorToolParams, URLValidatorToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        url: z.ZodOptional<z.ZodString>;
        urls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        checkSyntax: z.ZodDefault<z.ZodBoolean>;
        checkAccessibility: z.ZodDefault<z.ZodBoolean>;
        checkSuspicious: z.ZodDefault<z.ZodBoolean>;
        allowedProtocols: z.ZodDefault<z.ZodArray<z.ZodEnum<["http", "https", "ftp", "ftps", "mailto", "tel", "file"]>, "many">>;
        allowedDomains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        deniedDomains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        maxRedirects: z.ZodDefault<z.ZodNumber>;
        timeout: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        maxRedirects: number;
        checkSyntax: boolean;
        checkAccessibility: boolean;
        checkSuspicious: boolean;
        allowedProtocols: ("http" | "file" | "https" | "ftp" | "ftps" | "mailto" | "tel")[];
        url?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        urls?: string[] | undefined;
        allowedDomains?: string[] | undefined;
        deniedDomains?: string[] | undefined;
    }, {
        timeout?: number | undefined;
        url?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        urls?: string[] | undefined;
        maxRedirects?: number | undefined;
        checkSyntax?: boolean | undefined;
        allowedDomains?: string[] | undefined;
        deniedDomains?: string[] | undefined;
        checkAccessibility?: boolean | undefined;
        checkSuspicious?: boolean | undefined;
        allowedProtocols?: ("http" | "file" | "https" | "ftp" | "ftps" | "mailto" | "tel")[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        results: z.ZodArray<z.ZodObject<{
            url: z.ZodString;
            isValid: z.ZodBoolean;
            isAccessible: z.ZodOptional<z.ZodBoolean>;
            isSuspicious: z.ZodBoolean;
            protocol: z.ZodOptional<z.ZodString>;
            domain: z.ZodOptional<z.ZodString>;
            path: z.ZodOptional<z.ZodString>;
            query: z.ZodOptional<z.ZodString>;
            fragment: z.ZodOptional<z.ZodString>;
            statusCode: z.ZodOptional<z.ZodNumber>;
            redirectCount: z.ZodOptional<z.ZodNumber>;
            errors: z.ZodArray<z.ZodString, "many">;
            warnings: z.ZodArray<z.ZodString, "many">;
        }, "strip", z.ZodTypeAny, {
            url: string;
            errors: string[];
            isValid: boolean;
            warnings: string[];
            isSuspicious: boolean;
            path?: string | undefined;
            query?: string | undefined;
            domain?: string | undefined;
            statusCode?: number | undefined;
            protocol?: string | undefined;
            isAccessible?: boolean | undefined;
            fragment?: string | undefined;
            redirectCount?: number | undefined;
        }, {
            url: string;
            errors: string[];
            isValid: boolean;
            warnings: string[];
            isSuspicious: boolean;
            path?: string | undefined;
            query?: string | undefined;
            domain?: string | undefined;
            statusCode?: number | undefined;
            protocol?: string | undefined;
            isAccessible?: boolean | undefined;
            fragment?: string | undefined;
            redirectCount?: number | undefined;
        }>, "many">;
        stats: z.ZodObject<{
            totalURLs: z.ZodNumber;
            validURLs: z.ZodNumber;
            invalidURLs: z.ZodNumber;
            accessibleURLs: z.ZodNumber;
            suspiciousURLs: z.ZodNumber;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            totalURLs: number;
            validURLs: number;
            invalidURLs: number;
            accessibleURLs: number;
            suspiciousURLs: number;
        }, {
            processingTime: number;
            totalURLs: number;
            validURLs: number;
            invalidURLs: number;
            accessibleURLs: number;
            suspiciousURLs: number;
        }>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            totalURLs: number;
            validURLs: number;
            invalidURLs: number;
            accessibleURLs: number;
            suspiciousURLs: number;
        };
        results: {
            url: string;
            errors: string[];
            isValid: boolean;
            warnings: string[];
            isSuspicious: boolean;
            path?: string | undefined;
            query?: string | undefined;
            domain?: string | undefined;
            statusCode?: number | undefined;
            protocol?: string | undefined;
            isAccessible?: boolean | undefined;
            fragment?: string | undefined;
            redirectCount?: number | undefined;
        }[];
    }, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            totalURLs: number;
            validURLs: number;
            invalidURLs: number;
            accessibleURLs: number;
            suspiciousURLs: number;
        };
        results: {
            url: string;
            errors: string[];
            isValid: boolean;
            warnings: string[];
            isSuspicious: boolean;
            path?: string | undefined;
            query?: string | undefined;
            domain?: string | undefined;
            statusCode?: number | undefined;
            protocol?: string | undefined;
            isAccessible?: boolean | undefined;
            fragment?: string | undefined;
            redirectCount?: number | undefined;
        }[];
    }>;
    static readonly shortDescription = "Validate URLs with syntax, accessibility, and security checks";
    static readonly longDescription = "\n    A comprehensive URL validation tool with multiple verification layers.\n\n    Features:\n    - SYNTAX VALIDATION: RFC-compliant URL syntax checking\n    - ACCESSIBILITY CHECK: Verify URL is reachable (HTTP HEAD)\n    - SUSPICIOUS DETECTION: Identify malicious or dangerous patterns\n    - COMPONENT EXTRACTION: Parse URL into components\n    - DOMAIN VALIDATION: Whitelist/blacklist checking\n    - BATCH VALIDATION: Validate multiple URLs at once\n\n    Validation Checks:\n    - Syntax validation (RFC 3986)\n    - Protocol validation\n    - Domain accessibility (HTTP status)\n    - Suspicious pattern detection\n    - Domain whitelist/blacklist\n    - Redirect following\n    - Timeout enforcement\n\n    Security Checks:\n    - Directory traversal attempts\n    - XSS injection attempts\n    - URL encoding abuse\n    - Suspicious URL shorteners\n    - Credential injection attempts\n\n    Use cases:\n    - User-submitted URL validation\n    - Link quality checking\n    - Security scanning\n    - Web scraping preparation\n    - API endpoint validation\n    - Content moderation\n\n    Supported Protocols:\n    - HTTP, HTTPS (default)\n    - FTP, FTPS\n    - Mailto, Tel\n    - File\n\n    Accessibility Checks:\n    - HTTP HEAD request\n    - Status code verification\n    - Redirect following\n    - Timeout handling\n    - Error detection\n\n    Note: Accessibility checking requires network access.\n  ";
    static readonly alias = "url-validate";
    constructor(params: URLValidatorToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs URL validation
     */
    performAction(context?: BubbleContext): Promise<URLValidatorToolResult>;
    /**
     * Validate a single URL
     */
    private validateURL;
}
export {};
//# sourceMappingURL=url-validator-tool.d.ts.map