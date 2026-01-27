/**
 * Input Validator
 * Provides input validation and sanitization
 */
export declare class ValidationError extends Error {
    readonly errors: string[];
    constructor(message: string, errors: string[]);
}
export interface ValidationResult<T = any> {
    success: boolean;
    data?: T;
    errors?: string[];
}
export interface ValidationSchema {
    type: 'string' | 'number' | 'boolean' | 'object' | 'array';
    required?: boolean;
    minLength?: number;
    maxLength?: number;
    min?: number;
    max?: number;
    pattern?: RegExp;
    allowedValues?: any[];
    sanitize?: boolean;
}
export interface ObjectSchema {
    [key: string]: ValidationSchema;
}
/**
 * Sanitize a string input
 * @param input Raw input string
 * @returns Sanitized string
 */
export declare function sanitizeString(input: string): string;
/**
 * Validate and sanitize input against a schema
 * @param input Raw input
 * @param schema Validation schema
 * @returns ValidationResult
 */
export declare function validateInput<T = any>(input: any, schema: ValidationSchema | ObjectSchema): ValidationResult<T>;
/**
 * Validate that input is safe from code injection
 * @param input Input to check
 * @returns true if safe, false otherwise
 */
export declare function isSafeFromInjection(input: string): boolean;
/**
 * Validate and throw if invalid
 * @param input Input to validate
 * @param schema Validation schema
 * @throws ValidationError if validation fails
 */
export declare function validateOrThrow<T = any>(input: any, schema: ValidationSchema | ObjectSchema): T;
//# sourceMappingURL=input-validator.d.ts.map