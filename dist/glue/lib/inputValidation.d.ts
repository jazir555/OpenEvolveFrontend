/**
 * Input Validation - Security Configuration
 * Validates all user inputs to prevent injection attacks
 */
export interface ValidationResult {
    valid: boolean;
    errors: string[];
    sanitized?: any;
}
/**
 * Validate and sanitize string input
 */
export declare function validateString(input: string, fieldName: string, options?: {
    maxLength?: number;
    minLength?: number;
    pattern?: RegExp;
    allowEmpty?: boolean;
    trim?: boolean;
}): ValidationResult;
/**
 * Validate URL input
 */
export declare function validateUrl(input: string, fieldName: string): ValidationResult;
/**
 * Validate JSON input
 */
export declare function validateJson(input: string, fieldName: string): ValidationResult;
/**
 * Check for SQL injection
 */
export declare function checkSqlInjection(input: string, fieldName: string): ValidationResult;
/**
 * Check for XSS
 */
export declare function checkXss(input: string, fieldName: string): ValidationResult;
/**
 * Check for path traversal
 */
export declare function checkPathTraversal(input: string, fieldName: string): ValidationResult;
/**
 * Check for command injection
 */
export declare function checkCommandInjection(input: string, fieldName: string): ValidationResult;
/**
 * Comprehensive security validation
 * Runs all security checks
 */
export declare function validateSecurity(input: string, fieldName: string, options?: {
    checkSql?: boolean;
    checkXss?: boolean;
    checkPathTraversal?: boolean;
    checkCommandInjection?: boolean;
}): ValidationResult;
/**
 * Sanitize HTML output
 */
export declare function sanitizeHtml(input: string): string;
/**
 * Validate and sanitize user input completely
 */
export declare function validateUserInput(input: any, fieldName: string, validation?: {
    type?: 'string' | 'url' | 'json' | 'number';
    required?: boolean;
    securityChecks?: boolean;
    stringOptions?: {
        maxLength?: number;
        minLength?: number;
        pattern?: RegExp;
    };
}): ValidationResult;
//# sourceMappingURL=inputValidation.d.ts.map