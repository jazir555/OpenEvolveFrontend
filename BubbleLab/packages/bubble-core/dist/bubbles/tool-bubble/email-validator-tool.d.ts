/**
 * EMAIL VALIDATOR TOOL
 *
 * A tool bubble for comprehensive email validation including syntax checking,
 * domain verification, and disposable email detection.
 *
 * Features:
 * - Validate email syntax according to RFC standards
 * - Check domain MX records
 * - Detect disposable email addresses
 * - Check for role-based emails
 * - Suggest corrections for typos
 * - Batch email validation
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Email validator parameters schema
 */
declare const EmailValidatorToolParamsSchema: z.ZodObject<{
    email: z.ZodOptional<z.ZodString>;
    emails: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    checkSyntax: z.ZodDefault<z.ZodBoolean>;
    checkDomain: z.ZodDefault<z.ZodBoolean>;
    checkDisposable: z.ZodDefault<z.ZodBoolean>;
    checkRoleBased: z.ZodDefault<z.ZodBoolean>;
    suggestCorrections: z.ZodDefault<z.ZodBoolean>;
    allowedDomains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    deniedDomains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    checkSyntax: boolean;
    checkDomain: boolean;
    checkDisposable: boolean;
    checkRoleBased: boolean;
    suggestCorrections: boolean;
    email?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    emails?: string[] | undefined;
    allowedDomains?: string[] | undefined;
    deniedDomains?: string[] | undefined;
}, {
    email?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    emails?: string[] | undefined;
    checkSyntax?: boolean | undefined;
    checkDomain?: boolean | undefined;
    checkDisposable?: boolean | undefined;
    checkRoleBased?: boolean | undefined;
    suggestCorrections?: boolean | undefined;
    allowedDomains?: string[] | undefined;
    deniedDomains?: string[] | undefined;
}>;
/**
 * Email validator result schema
 */
declare const EmailValidatorToolResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    results: z.ZodArray<z.ZodObject<{
        email: z.ZodString;
        isValid: z.ZodBoolean;
        isDisposable: z.ZodBoolean;
        isRoleBased: z.ZodBoolean;
        domain: z.ZodString;
        syntaxValid: z.ZodBoolean;
        domainValid: z.ZodOptional<z.ZodBoolean>;
        suggestions: z.ZodArray<z.ZodString, "many">;
        errors: z.ZodArray<z.ZodString, "many">;
    }, "strip", z.ZodTypeAny, {
        email: string;
        domain: string;
        errors: string[];
        isValid: boolean;
        isDisposable: boolean;
        isRoleBased: boolean;
        syntaxValid: boolean;
        suggestions: string[];
        domainValid?: boolean | undefined;
    }, {
        email: string;
        domain: string;
        errors: string[];
        isValid: boolean;
        isDisposable: boolean;
        isRoleBased: boolean;
        syntaxValid: boolean;
        suggestions: string[];
        domainValid?: boolean | undefined;
    }>, "many">;
    stats: z.ZodObject<{
        totalEmails: z.ZodNumber;
        validEmails: z.ZodNumber;
        invalidEmails: z.ZodNumber;
        disposableEmails: z.ZodNumber;
        roleBasedEmails: z.ZodNumber;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        totalEmails: number;
        validEmails: number;
        invalidEmails: number;
        disposableEmails: number;
        roleBasedEmails: number;
    }, {
        processingTime: number;
        totalEmails: number;
        validEmails: number;
        invalidEmails: number;
        disposableEmails: number;
        roleBasedEmails: number;
    }>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        totalEmails: number;
        validEmails: number;
        invalidEmails: number;
        disposableEmails: number;
        roleBasedEmails: number;
    };
    results: {
        email: string;
        domain: string;
        errors: string[];
        isValid: boolean;
        isDisposable: boolean;
        isRoleBased: boolean;
        syntaxValid: boolean;
        suggestions: string[];
        domainValid?: boolean | undefined;
    }[];
}, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        totalEmails: number;
        validEmails: number;
        invalidEmails: number;
        disposableEmails: number;
        roleBasedEmails: number;
    };
    results: {
        email: string;
        domain: string;
        errors: string[];
        isValid: boolean;
        isDisposable: boolean;
        isRoleBased: boolean;
        syntaxValid: boolean;
        suggestions: string[];
        domainValid?: boolean | undefined;
    }[];
}>;
type EmailValidatorToolParams = z.output<typeof EmailValidatorToolParamsSchema>;
type EmailValidatorToolResult = z.output<typeof EmailValidatorToolResultSchema>;
type EmailValidatorToolParamsInput = z.input<typeof EmailValidatorToolParamsSchema>;
/**
 * Email Validator Tool
 * Comprehensive email validation with advanced checks
 */
export declare class EmailValidatorTool extends ToolBubble<EmailValidatorToolParams, EmailValidatorToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        email: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        checkSyntax: z.ZodDefault<z.ZodBoolean>;
        checkDomain: z.ZodDefault<z.ZodBoolean>;
        checkDisposable: z.ZodDefault<z.ZodBoolean>;
        checkRoleBased: z.ZodDefault<z.ZodBoolean>;
        suggestCorrections: z.ZodDefault<z.ZodBoolean>;
        allowedDomains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        deniedDomains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        checkSyntax: boolean;
        checkDomain: boolean;
        checkDisposable: boolean;
        checkRoleBased: boolean;
        suggestCorrections: boolean;
        email?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        emails?: string[] | undefined;
        allowedDomains?: string[] | undefined;
        deniedDomains?: string[] | undefined;
    }, {
        email?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        emails?: string[] | undefined;
        checkSyntax?: boolean | undefined;
        checkDomain?: boolean | undefined;
        checkDisposable?: boolean | undefined;
        checkRoleBased?: boolean | undefined;
        suggestCorrections?: boolean | undefined;
        allowedDomains?: string[] | undefined;
        deniedDomains?: string[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        results: z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            isValid: z.ZodBoolean;
            isDisposable: z.ZodBoolean;
            isRoleBased: z.ZodBoolean;
            domain: z.ZodString;
            syntaxValid: z.ZodBoolean;
            domainValid: z.ZodOptional<z.ZodBoolean>;
            suggestions: z.ZodArray<z.ZodString, "many">;
            errors: z.ZodArray<z.ZodString, "many">;
        }, "strip", z.ZodTypeAny, {
            email: string;
            domain: string;
            errors: string[];
            isValid: boolean;
            isDisposable: boolean;
            isRoleBased: boolean;
            syntaxValid: boolean;
            suggestions: string[];
            domainValid?: boolean | undefined;
        }, {
            email: string;
            domain: string;
            errors: string[];
            isValid: boolean;
            isDisposable: boolean;
            isRoleBased: boolean;
            syntaxValid: boolean;
            suggestions: string[];
            domainValid?: boolean | undefined;
        }>, "many">;
        stats: z.ZodObject<{
            totalEmails: z.ZodNumber;
            validEmails: z.ZodNumber;
            invalidEmails: z.ZodNumber;
            disposableEmails: z.ZodNumber;
            roleBasedEmails: z.ZodNumber;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            totalEmails: number;
            validEmails: number;
            invalidEmails: number;
            disposableEmails: number;
            roleBasedEmails: number;
        }, {
            processingTime: number;
            totalEmails: number;
            validEmails: number;
            invalidEmails: number;
            disposableEmails: number;
            roleBasedEmails: number;
        }>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            totalEmails: number;
            validEmails: number;
            invalidEmails: number;
            disposableEmails: number;
            roleBasedEmails: number;
        };
        results: {
            email: string;
            domain: string;
            errors: string[];
            isValid: boolean;
            isDisposable: boolean;
            isRoleBased: boolean;
            syntaxValid: boolean;
            suggestions: string[];
            domainValid?: boolean | undefined;
        }[];
    }, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            totalEmails: number;
            validEmails: number;
            invalidEmails: number;
            disposableEmails: number;
            roleBasedEmails: number;
        };
        results: {
            email: string;
            domain: string;
            errors: string[];
            isValid: boolean;
            isDisposable: boolean;
            isRoleBased: boolean;
            syntaxValid: boolean;
            suggestions: string[];
            domainValid?: boolean | undefined;
        }[];
    }>;
    static readonly shortDescription = "Validate email addresses with syntax, domain, and quality checks";
    static readonly longDescription = "\n    A comprehensive email validation tool with multiple verification layers.\n\n    Features:\n    - SYNTAX VALIDATION: RFC-compliant email syntax checking\n    - DOMAIN VERIFICATION: Check if domain has MX records\n    - DISPOSABLE DETECTION: Identify temporary email addresses\n    - ROLE-BASED DETECTION: Identify generic role addresses\n    - TYPO CORRECTION: Suggest corrections for common mistakes\n    - BATCH VALIDATION: Validate multiple emails at once\n\n    Validation Checks:\n    - Syntax validation (RFC 5322)\n    - Domain existence (MX records)\n    - Disposable email detection\n    - Role-based address detection\n    - Domain whitelist/blacklist\n    - Common typo detection\n\n    Use cases:\n    - Email list cleaning\n    - User registration validation\n    - Lead qualification\n    - Fraud prevention\n    - Email marketing quality control\n    - Contact form validation\n\n    Disposable Email Domains Detected:\n    - tempmail.com, guerrillamail.com, mailinator.com\n    - 10minutemail.com, yopmail.com, trashmail.com\n    - And many more\n\n    Role-Based Addresses Detected:\n    - admin@, support@, info@, sales@\n    - marketing@, contact@, office@, billing@\n    - And many more\n\n    Common Typos Corrected:\n    - gmial.com -> gmail.com\n    - yahooo.com -> yahoo.com\n    - hotmial.com -> hotmail.com\n    - And many more\n\n    Note: Domain MX record checking requires DNS lookup capability.\n  ";
    static readonly alias = "email-validate";
    constructor(params: EmailValidatorToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs email validation
     */
    performAction(context?: BubbleContext): Promise<EmailValidatorToolResult>;
    /**
     * Validate a single email
     */
    private validateEmail;
    /**
     * Check MX records for a domain using DNS lookup
     */
    private checkMXRecords;
    /**
     * Check for common TLD typos
     */
    private checkTLDTypos;
}
export {};
//# sourceMappingURL=email-validator-tool.d.ts.map