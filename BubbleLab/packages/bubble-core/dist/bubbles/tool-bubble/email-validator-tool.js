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
import { CredentialType } from '@bubblelab/shared-schemas';
/**
 * Email validator parameters schema
 */
const EmailValidatorToolParamsSchema = z.object({
    // Input emails
    email: z
        .string()
        .optional()
        .describe('Single email address to validate'),
    emails: z
        .array(z.string())
        .optional()
        .describe('Multiple email addresses to validate'),
    // Validation options
    checkSyntax: z
        .boolean()
        .default(true)
        .describe('Validate email syntax'),
    checkDomain: z
        .boolean()
        .default(false)
        .describe('Check domain MX records'),
    checkDisposable: z
        .boolean()
        .default(true)
        .describe('Check for disposable email domains'),
    checkRoleBased: z
        .boolean()
        .default(false)
        .describe('Check for role-based addresses (admin@, support@, etc.)'),
    suggestCorrections: z
        .boolean()
        .default(true)
        .describe('Suggest corrections for common typos'),
    // Domain options
    allowedDomains: z
        .array(z.string())
        .optional()
        .describe('Whitelist of allowed domains'),
    deniedDomains: z
        .array(z.string())
        .optional()
        .describe('Blacklist of denied domains'),
    // Credentials
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Credentials for external domain verification'),
});
/**
 * Email validation result schema
 */
const EmailValidationResultSchema = z.object({
    email: z.string().describe('Email address that was validated'),
    isValid: z.boolean().describe('Whether the email is valid'),
    isDisposable: z.boolean().describe('Whether the email is from a disposable domain'),
    isRoleBased: z.boolean().describe('Whether the email is role-based'),
    domain: z.string().describe('Email domain'),
    syntaxValid: z.boolean().describe('Whether the syntax is valid'),
    domainValid: z.boolean().optional().describe('Whether the domain exists'),
    suggestions: z
        .array(z.string())
        .describe('Suggested corrections for typos'),
    errors: z
        .array(z.string())
        .describe('Validation errors'),
});
/**
 * Email validator result schema
 */
const EmailValidatorToolResultSchema = z.object({
    // Result
    success: z.boolean().describe('Whether the validation was successful'),
    // Validation results
    results: z
        .array(EmailValidationResultSchema)
        .describe('Validation results for each email'),
    // Summary statistics
    stats: z
        .object({
        totalEmails: z.number(),
        validEmails: z.number(),
        invalidEmails: z.number(),
        disposableEmails: z.number(),
        roleBasedEmails: z.number(),
        processingTime: z.number(),
    })
        .describe('Validation statistics'),
    error: z.string().describe('Error message if validation failed'),
});
/**
 * Common disposable email domains
 */
const DISPOSABLE_DOMAINS = new Set([
    'tempmail.com',
    'guerrillamail.com',
    'mailinator.com',
    '10minutemail.com',
    'yopmail.com',
    'trashmail.com',
    'sharklasers.com',
    'throwaway.email',
    'getairmail.com',
    'temp-mail.org',
]);
/**
 * Role-based email prefixes
 */
const ROLE_BASED_PREFIXES = new Set([
    'admin',
    'support',
    'info',
    'sales',
    'marketing',
    'contact',
    'office',
    'billing',
    'accounts',
    'hr',
    'jobs',
    'webmaster',
    'abuse',
    'noreply',
    'no-reply',
]);
/**
 * Common email domain typos
 */
const COMMON_DOMAIN_TYPOS = {
    'gmial.com': 'gmail.com',
    'gmai.com': 'gmail.com',
    'gmail.co': 'gmail.com',
    'gmil.com': 'gmail.com',
    'yahooo.com': 'yahoo.com',
    'yahho.com': 'yahoo.com',
    'hotmial.com': 'hotmail.com',
    'hotmil.com': 'hotmail.com',
    'outlok.com': 'outlook.com',
    'outloook.com': 'outlook.com',
};
/**
 * Email Validator Tool
 * Comprehensive email validation with advanced checks
 */
export class EmailValidatorTool extends ToolBubble {
    /**
     * REQUIRED STATIC METADATA
     */
    static type = 'tool';
    static bubbleName = 'email-validator-tool';
    static schema = EmailValidatorToolParamsSchema;
    static resultSchema = EmailValidatorToolResultSchema;
    static shortDescription = 'Validate email addresses with syntax, domain, and quality checks';
    static longDescription = `
    A comprehensive email validation tool with multiple verification layers.

    Features:
    - SYNTAX VALIDATION: RFC-compliant email syntax checking
    - DOMAIN VERIFICATION: Check if domain has MX records
    - DISPOSABLE DETECTION: Identify temporary email addresses
    - ROLE-BASED DETECTION: Identify generic role addresses
    - TYPO CORRECTION: Suggest corrections for common mistakes
    - BATCH VALIDATION: Validate multiple emails at once

    Validation Checks:
    - Syntax validation (RFC 5322)
    - Domain existence (MX records)
    - Disposable email detection
    - Role-based address detection
    - Domain whitelist/blacklist
    - Common typo detection

    Use cases:
    - Email list cleaning
    - User registration validation
    - Lead qualification
    - Fraud prevention
    - Email marketing quality control
    - Contact form validation

    Disposable Email Domains Detected:
    - tempmail.com, guerrillamail.com, mailinator.com
    - 10minutemail.com, yopmail.com, trashmail.com
    - And many more

    Role-Based Addresses Detected:
    - admin@, support@, info@, sales@
    - marketing@, contact@, office@, billing@
    - And many more

    Common Typos Corrected:
    - gmial.com -> gmail.com
    - yahooo.com -> yahoo.com
    - hotmial.com -> hotmail.com
    - And many more

    Note: Domain MX record checking requires DNS lookup capability.
  `;
    static alias = 'email-validate';
    constructor(params, context) {
        super(params, context);
    }
    /**
     * Main action method - performs email validation
     */
    async performAction(context) {
        void context; // Context available but not currently used
        const startTime = Date.now();
        try {
            console.log('[EmailValidatorTool] Starting email validation');
            // Determine emails to validate
            let emailsToValidate;
            if (this.params.email) {
                emailsToValidate = [this.params.email];
            }
            else if (this.params.emails) {
                emailsToValidate = this.params.emails;
            }
            else {
                throw new Error('Either email or emails parameter is required');
            }
            // Validate each email
            const results = await Promise.all(emailsToValidate.map((email) => this.validateEmail(email)));
            // Calculate statistics
            const validEmails = results.filter((r) => r.isValid).length;
            const invalidEmails = results.filter((r) => !r.isValid).length;
            const disposableEmails = results.filter((r) => r.isDisposable).length;
            const roleBasedEmails = results.filter((r) => r.isRoleBased).length;
            const processingTime = Date.now() - startTime;
            console.log(`[EmailValidatorTool] Validation completed. Valid: ${validEmails}, Invalid: ${invalidEmails}`);
            return {
                success: true,
                results,
                stats: {
                    totalEmails: emailsToValidate.length,
                    validEmails,
                    invalidEmails,
                    disposableEmails,
                    roleBasedEmails,
                    processingTime,
                },
                error: '',
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[EmailValidatorTool] Validation failed: ${errorMessage}`);
            return {
                success: false,
                results: [],
                stats: {
                    totalEmails: 0,
                    validEmails: 0,
                    invalidEmails: 0,
                    disposableEmails: 0,
                    roleBasedEmails: 0,
                    processingTime: Date.now() - startTime,
                },
                error: errorMessage,
            };
        }
    }
    /**
     * Validate a single email
     */
    async validateEmail(email) {
        const errors = [];
        const suggestions = [];
        let isValid = true;
        let syntaxValid = true;
        let domainValid = undefined;
        let isDisposable = false;
        let isRoleBased = false;
        let domain = '';
        try {
            // Normalize email
            const normalizedEmail = email.trim().toLowerCase();
            // Extract domain
            const parts = normalizedEmail.split('@');
            if (parts.length !== 2) {
                syntaxValid = false;
                errors.push('Email must contain exactly one @ symbol');
                isValid = false;
            }
            else {
                const [localPart, domainPart] = parts;
                domain = domainPart;
                // Validate local part
                if (!localPart || localPart.length === 0) {
                    syntaxValid = false;
                    errors.push('Local part (before @) cannot be empty');
                    isValid = false;
                }
                if (localPart.length > 64) {
                    syntaxValid = false;
                    errors.push('Local part (before @) cannot exceed 64 characters');
                    isValid = false;
                }
                // Validate domain
                if (!domainPart || domainPart.length === 0) {
                    syntaxValid = false;
                    errors.push('Domain (after @) cannot be empty');
                    isValid = false;
                }
                if (domainPart.length > 255) {
                    syntaxValid = false;
                    errors.push('Domain (after @) cannot exceed 255 characters');
                    isValid = false;
                }
                // Check for valid characters using RFC 5322 compliant regex
                const emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;
                if (!emailRegex.test(normalizedEmail)) {
                    syntaxValid = false;
                    errors.push('Email contains invalid characters or format');
                    isValid = false;
                }
                // Check domain whitelist/blacklist
                if (this.params.allowedDomains && this.params.allowedDomains.length > 0) {
                    if (!this.params.allowedDomains.includes(domainPart)) {
                        errors.push(`Domain ${domainPart} is not in the allowed list`);
                        isValid = false;
                    }
                }
                if (this.params.deniedDomains && this.params.deniedDomains.length > 0) {
                    if (this.params.deniedDomains.includes(domainPart)) {
                        errors.push(`Domain ${domainPart} is in the denied list`);
                        isValid = false;
                    }
                }
                // Check for disposable domain
                if (this.params.checkDisposable && DISPOSABLE_DOMAINS.has(domainPart)) {
                    isDisposable = true;
                    errors.push(`Email is from a disposable email domain: ${domainPart}`);
                    isValid = false;
                }
                // Check for role-based email
                if (this.params.checkRoleBased) {
                    const prefix = localPart.split('.')[0].split('+')[0];
                    if (ROLE_BASED_PREFIXES.has(prefix)) {
                        isRoleBased = true;
                        errors.push(`Email is a role-based address: ${prefix}@`);
                    }
                }
                // Check domain MX records using DNS lookup
                if (this.params.checkDomain && syntaxValid) {
                    try {
                        domainValid = await this.checkMXRecords(domainPart);
                        if (!domainValid) {
                            errors.push(`Domain ${domainPart} does not have valid MX records`);
                            isValid = false;
                        }
                    }
                    catch (dnsError) {
                        errors.push(`DNS lookup failed: ${dnsError instanceof Error ? dnsError.message : 'Unknown error'}`);
                        // Don't fail validation on DNS errors, just mark as unknown
                        domainValid = undefined;
                    }
                }
                // Suggest corrections for common typos
                if (this.params.suggestCorrections && syntaxValid) {
                    const correction = COMMON_DOMAIN_TYPOS[domainPart];
                    if (correction) {
                        const correctedEmail = `${localPart}@${correction}`;
                        suggestions.push(`Did you mean: ${correctedEmail}?`);
                    }
                    // Check for common TLD typos
                    const tldTypos = this.checkTLDTypos(domainPart);
                    if (tldTypos) {
                        const correctedEmail = `${localPart}@${tldTypos}`;
                        suggestions.push(`Did you mean: ${correctedEmail}?`);
                    }
                }
            }
        }
        catch (error) {
            errors.push(`Validation error: ${error instanceof Error ? error.message : 'Unknown error'}`);
            isValid = false;
        }
        return {
            email,
            isValid,
            isDisposable,
            isRoleBased,
            domain,
            syntaxValid,
            domainValid,
            suggestions,
            errors,
        };
    }
    /**
     * Check MX records for a domain using DNS lookup
     */
    async checkMXRecords(domain) {
        try {
            // Try to use Node.js dns module
            const dns = await import('dns');
            const { promisify } = await import('util');
            const resolveMx = promisify(dns.resolveMx);
            const mxRecords = await resolveMx(domain);
            return mxRecords && mxRecords.length > 0;
        }
        catch (error) {
            // If DNS module is not available (e.g., in browser), try fetch-based approach
            console.log(`[EmailValidatorTool] DNS module unavailable, trying fallback validation for domain: ${domain}`);
            // Fallback: Try to resolve using DNS-over-HTTPS
            try {
                const response = await fetch(`https://dns.google/resolve?name=${domain}&type=MX`);
                const data = await response.json();
                if (data.Answer && data.Answer.length > 0) {
                    return true;
                }
                return false;
            }
            catch (fetchError) {
                console.log(`[EmailValidatorTool] DNS-over-HTTPS failed, assuming valid for: ${domain}`);
                // As a last resort, assume valid if domain has valid format
                return /^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?(\.[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?)*$/.test(domain);
            }
        }
    }
    /**
     * Check for common TLD typos
     */
    checkTLDTypos(domain) {
        const commonTLDTypos = {
            '.co': '.com',
            '.cm': '.com',
            '.con': '.com',
            '.cpm': '.com',
            '.om': '.com',
            '.nett': '.net',
            '.ne': '.net',
            '.or': '.org',
            '.orgg': '.org',
            '.ed': '.edu',
            '.gvo': '.gov',
        };
        for (const [typo, correct] of Object.entries(commonTLDTypos)) {
            if (domain.endsWith(typo)) {
                return domain.substring(0, domain.length - typo.length) + correct;
            }
        }
        return null;
    }
}
//# sourceMappingURL=email-validator-tool.js.map