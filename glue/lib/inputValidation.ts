/**
 * Input Validation - Security Configuration
 * Validates all user inputs to prevent injection attacks
 */

import { logger } from './structuredLogger';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  sanitized?: any;
}

/**
 * SQL Injection detection patterns
 */
const SQL_INJECTION_PATTERNS = [
  /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)/i,
  /(--)|(\/\*)|(\*\/)/,
  /(\bor\b|\band\b).*?=/i,
  /(\bwaitfor\b\s+delay\b)/i,
  /(;|\s+)(exec|execute)\s/i,
  /('.*?--)/,
  /(\bxp_|sp_)\w+/i // SQL Server system procs
];

/**
 * XSS detection patterns
 */
const XSS_PATTERNS = [
  /<script[^>]*>.*?<\/script>/gi,
  /<iframe[^>]*>.*?<\/iframe>/gi,
  /javascript:/gi,
  /on\w+\s*=/gi, // onclick=, onload=, etc.
  /<[^>]*on\w+\s*=/gi,
  /<[^>]*style[^>]*>/gi
];

/**
 * Path traversal detection
 */
const PATH_TRAVERSAL_PATTERNS = [
  /\.\.[\/\\]/,
  /%2e%2e[\/\\%]/i,
  /[\/\\]\.\.[\/\\]/
];

/**
 * Command injection detection
 */
const COMMAND_INJECTION_PATTERNS = [
  /[;&|`$()]/,
  /\|\|/,
  /&&/,
  /`.*?\$.*?`/
];

/**
 * Validate and sanitize string input
 */
export function validateString(
  input: string,
  fieldName: string,
  options: {
    maxLength?: number;
    minLength?: number;
    pattern?: RegExp;
    allowEmpty?: boolean;
    trim?: boolean;
  } = {}
): ValidationResult {
  const errors: string[] = [];

  let value = input;
  if (options.trim !== false) {
    value = value.trim();
  }

  // Check empty
  if (!options.allowEmpty && value.length === 0) {
    errors.push(`${fieldName} cannot be empty`);
  }

  // Check length
  if (options.minLength && value.length < options.minLength) {
    errors.push(`${fieldName} must be at least ${options.minLength} characters`);
  }

  if (options.maxLength && value.length > options.maxLength) {
    errors.push(`${fieldName} must be at most ${options.maxLength} characters`);
  }

  // Check pattern
  if (options.pattern && !options.pattern.test(value)) {
    errors.push(`${fieldName} format is invalid`);
  }

  return {
    valid: errors.length === 0,
    errors,
    sanitized: value
  };
}

/**
 * Validate URL input
 */
export function validateUrl(input: string, fieldName: string): ValidationResult {
  const errors: string[] = [];

  try {
    const url = new URL(input);

    // Only allow http/https
    if (!['http:', 'https:'].includes(url.protocol)) {
      errors.push(`${fieldName} must use HTTP or HTTPS protocol`);
    }

    // Prevent localhost in production
    if (process.env.NODE_ENV === 'production'
        && (url.hostname === 'localhost' || url.hostname === '127.0.0.1')) {
      errors.push(`${fieldName} cannot point to localhost in production`);
    }

    return {
      valid: errors.length === 0,
      errors,
      sanitized: url.toString()
    };
  } catch (error) {
    return {
      valid: false,
      errors: [`${fieldName} is not a valid URL`]
    };
  }
}

/**
 * Validate JSON input
 */
export function validateJson(input: string, fieldName: string): ValidationResult {
  const errors: string[] = [];

  try {
    const parsed = JSON.parse(input);

    if (typeof parsed !== 'object' || parsed === null) {
      errors.push(`${fieldName} must be a JSON object`);
    }

    return {
      valid: errors.length === 0,
      errors,
      sanitized: parsed
    };
  } catch (error) {
    return {
      valid: false,
      errors: [`${fieldName} is not valid JSON`]
    };
  }
}

/**
 * Check for SQL injection
 */
export function checkSqlInjection(input: string, fieldName: string): ValidationResult {
  const errors: string[] = [];

  for (const pattern of SQL_INJECTION_PATTERNS) {
    if (pattern.test(input)) {
      errors.push(`${fieldName} contains potential SQL injection: ${input}`);
      logger.warn(`SQL injection attempt detected`, {
        field: fieldName,
        input: input.substring(0, 100) // Log only first 100 chars
      });
      break;
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Check for XSS
 */
export function checkXss(input: string, fieldName: string): ValidationResult {
  const errors: string[] = [];

  for (const pattern of XSS_PATTERNS) {
    if (pattern.test(input)) {
      errors.push(`${fieldName} contains potential XSS: ${input}`);
      logger.warn(`XSS attempt detected`, {
        field: fieldName,
        input: input.substring(0, 100)
      });
      break;
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Check for path traversal
 */
export function checkPathTraversal(input: string, fieldName: string): ValidationResult {
  const errors: string[] = [];

  for (const pattern of PATH_TRAVERSAL_PATTERNS) {
    if (pattern.test(input)) {
      errors.push(`${fieldName} contains path traversal attempt`);
      logger.warn(`Path traversal attempt detected`, {
        field: fieldName,
        input: input.substring(0, 100)
      });
      break;
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Check for command injection
 */
export function checkCommandInjection(input: string, fieldName: string): ValidationResult {
  const errors: string[] = [];

  for (const pattern of COMMAND_INJECTION_PATTERNS) {
    if (pattern.test(input)) {
      errors.push(`${fieldName} contains command injection attempt`);
      logger.warn(`Command injection attempt detected`, {
        field: fieldName,
        input: input.substring(0, 100)
      });
      break;
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Comprehensive security validation
 * Runs all security checks
 */
export function validateSecurity(
  input: string,
  fieldName: string,
  options: {
    checkSql?: boolean;
    checkXss?: boolean;
    checkPathTraversal?: boolean;
    checkCommandInjection?: boolean;
  } = {}
): ValidationResult {
  const errors: string[] = [];

  // Enable all checks by default for user input
  const checks = {
    checkSql: true,
    checkXss: true,
    checkPathTraversal: true,
    checkCommandInjection: true,
    ...options
  };

  if (checks.checkSql) {
    const sqlResult = checkSqlInjection(input, fieldName);
    errors.push(...sqlResult.errors);
  }

  if (checks.checkXss) {
    const xssResult = checkXss(input, fieldName);
    errors.push(...xssResult.errors);
  }

  if (checks.checkPathTraversal) {
    const pathResult = checkPathTraversal(input, fieldName);
    errors.push(...pathResult.errors);
  }

  if (checks.checkCommandInjection) {
    const cmdResult = checkCommandInjection(input, fieldName);
    errors.push(...cmdResult.errors);
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Sanitize HTML output
 */
export function sanitizeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
}

/**
 * Validate and sanitize user input completely
 */
export function validateUserInput(
  input: any,
  fieldName: string,
  validation: {
    type?: 'string' | 'url' | 'json' | 'number';
    required?: boolean;
    securityChecks?: boolean;
    stringOptions?: {
      maxLength?: number;
      minLength?: number;
      pattern?: RegExp;
    };
  } = {}
): ValidationResult {
  const errors: string[] = [];
  let sanitized: any;

  // Check required
  if (validation.required && (input === undefined || input === null || input === '')) {
    errors.push(`${fieldName} is required`);
    return { valid: false, errors };
  }

  // Skip validation if not required and empty
  if (!validation.required && (input === undefined || input === null || input === '')) {
    return { valid: true, errors: [], sanitized: input };
  }

  // Type-specific validation
  switch (validation.type) {
    case 'string':
      const stringResult = validateString(input, fieldName, validation.stringOptions);
      errors.push(...stringResult.errors);
      sanitized = stringResult.sanitized;
      break;

    case 'url':
      const urlResult = validateUrl(input, fieldName);
      errors.push(...urlResult.errors);
      sanitized = urlResult.sanitized;
      break;

    case 'json':
      const jsonResult = validateJson(input, fieldName);
      errors.push(...jsonResult.errors);
      sanitized = jsonResult.sanitized;
      break;

    case 'number':
      const num = Number(input);
      if (isNaN(num)) {
        errors.push(`${fieldName} must be a number`);
      } else {
        sanitized = num;
      }
      break;

    default:
      sanitized = input;
  }

  // Security validation for strings
  if (validation.securityChecks !== false && typeof sanitized === 'string') {
    const securityResult = validateSecurity(sanitized, fieldName);
    errors.push(...securityResult.errors);
  }

  return {
    valid: errors.length === 0,
    errors,
    sanitized
  };
}
