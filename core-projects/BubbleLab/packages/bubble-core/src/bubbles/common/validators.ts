/**
 * Common validation utilities for Bubble implementations
 * Provides reusable validation functions to ensure data integrity and security
 */

import { z } from 'zod';

/**
 * Email validation regex
 * Follows RFC 5322 standard with practical simplifications
 */
export const EMAIL_REGEX = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;

/**
 * URL validation regex
 * Allows http, https protocols
 */
export const URL_REGEX = /^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$/;

/**
 * ISO 8601 timestamp validation regex
 */
export const ISO_TIMESTAMP_REGEX = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$/;

/**
 * Common validation error class
 */
export class ValidationError extends Error {
  constructor(message: string, public field?: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

/**
 * Validate email address
 * @param email - Email address to validate
 * @returns True if email is valid
 * @throws ValidationError if email is invalid
 */
export function validateEmail(email: string): boolean {
  if (!email || typeof email !== 'string') {
    throw new ValidationError('Email is required and must be a string', 'email');
  }

  if (!EMAIL_REGEX.test(email)) {
    throw new ValidationError('Invalid email format', 'email');
  }

  if (email.length > 254) { // RFC 5321 limit
    throw new ValidationError('Email address exceeds maximum length of 254 characters', 'email');
  }

  return true;
}

/**
 * Validate URL
 * @param url - URL to validate
 * @param allowedProtocols - Array of allowed protocols (default: ['http', 'https'])
 * @returns True if URL is valid
 * @throws ValidationError if URL is invalid
 */
export function validateUrl(url: string, allowedProtocols: string[] = ['http', 'https']): boolean {
  if (!url || typeof url !== 'string') {
    throw new ValidationError('URL is required and must be a string', 'url');
  }

  try {
    const parsedUrl = new URL(url);

    if (!allowedProtocols.includes(parsedUrl.protocol.replace(':', ''))) {
      throw new ValidationError(
        `URL protocol must be one of: ${allowedProtocols.join(', ')}`,
        'url'
      );
    }

    return true;
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }
    throw new ValidationError('Invalid URL format', 'url');
  }
}

/**
 * Validate timestamp in ISO 8601 format
 * @param timestamp - Timestamp string to validate
 * @returns True if timestamp is valid
 * @throws ValidationError if timestamp is invalid
 */
export function validateTimestamp(timestamp: string): boolean {
  if (!timestamp || typeof timestamp !== 'string') {
    throw new ValidationError('Timestamp is required and must be a string', 'timestamp');
  }

  if (!ISO_TIMESTAMP_REGEX.test(timestamp)) {
    throw new ValidationError('Timestamp must be in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)', 'timestamp');
  }

  // Verify it's a valid date
  const date = new Date(timestamp);
  if (isNaN(date.getTime())) {
    throw new ValidationError('Invalid date value', 'timestamp');
  }

  return true;
}

/**
 * Validate that a string is not empty or just whitespace
 * @param value - String value to validate
 * @param fieldName - Name of the field for error messages
 * @returns True if value is valid
 * @throws ValidationError if value is empty
 */
export function validateNonEmptyString(value: string, fieldName: string = 'value'): boolean {
  if (!value || typeof value !== 'string') {
    throw new ValidationError(`${fieldName} is required and must be a string`, fieldName);
  }

  if (value.trim().length === 0) {
    throw new ValidationError(`${fieldName} cannot be empty or whitespace`, fieldName);
  }

  return true;
}

/**
 * Validate number is within range
 * @param value - Number to validate
 * @param min - Minimum allowed value (inclusive)
 * @param max - Maximum allowed value (inclusive)
 * @param fieldName - Name of the field for error messages
 * @returns True if value is within range
 * @throws ValidationError if value is out of range
 */
export function validateNumberRange(
  value: number,
  min: number,
  max: number,
  fieldName: string = 'value'
): boolean {
  if (typeof value !== 'number' || isNaN(value)) {
    throw new ValidationError(`${fieldName} must be a valid number`, fieldName);
  }

  if (value < min || value > max) {
    throw new ValidationError(
      `${fieldName} must be between ${min} and ${max} (inclusive)`,
      fieldName
    );
  }

  return true;
}

/**
 * Validate array length
 * @param array - Array to validate
 * @param minLength - Minimum allowed length (default: 0)
 * @param maxLength - Maximum allowed length
 * @param fieldName - Name of the field for error messages
 * @returns True if array length is valid
 * @throws ValidationError if array length is invalid
 */
export function validateArrayLength<T>(
  array: T[],
  minLength: number = 0,
  maxLength: number,
  fieldName: string = 'array'
): boolean {
  if (!Array.isArray(array)) {
    throw new ValidationError(`${fieldName} must be an array`, fieldName);
  }

  if (array.length < minLength) {
    throw new ValidationError(
      `${fieldName} must contain at least ${minLength} item(s)`,
      fieldName
    );
  }

  if (array.length > maxLength) {
    throw new ValidationError(
      `${fieldName} cannot contain more than ${maxLength} item(s)`,
      fieldName
    );
  }

  return true;
}

/**
 * Validate object has required properties
 * @param obj - Object to validate
 * @param requiredProps - Array of required property names
 * @param fieldName - Name of the field for error messages
 * @returns True if object has all required properties
 * @throws ValidationError if object is missing required properties
 */
export function validateRequiredProperties(
  obj: Record<string, unknown>,
  requiredProps: string[],
  fieldName: string = 'object'
): boolean {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
    throw new ValidationError(`${fieldName} must be an object`, fieldName);
  }

  const missingProps = requiredProps.filter(prop => !(prop in obj) || obj[prop] === undefined);

  if (missingProps.length > 0) {
    throw new ValidationError(
      `${fieldName} is missing required properties: ${missingProps.join(', ')}`,
      fieldName
    );
  }

  return true;
}

/**
 * Sanitize string input to prevent injection attacks
 * @param input - String to sanitize
 * @returns Sanitized string
 */
export function sanitizeString(input: string): string {
  if (typeof input !== 'string') {
    return '';
  }

  return input
    .replace(/[<>]/g, '') // Remove potential HTML/XML tags
    .replace(/[()]/g, '') // Remove parentheses (XSS prevention)
    .replace(/['"]/g, '') // Remove quotes
    .replace(/;/g, '') // Remove semicolons (SQL injection prevention)
    .replace(/--/g, '') // Remove SQL comment markers
    .trim();
}

/**
 * Validate and sanitize file path to prevent path traversal
 * @param filePath - File path to validate
 * @param allowAbsolutePaths - Whether to allow absolute paths (default: false)
 * @returns True if path is valid
 * @throws ValidationError if path contains dangerous patterns
 */
export function validateFilePath(filePath: string, allowAbsolutePaths: boolean = false): boolean {
  if (!filePath || typeof filePath !== 'string') {
    throw new ValidationError('File path is required and must be a string', 'filePath');
  }

  // Block path traversal attempts
  if (filePath.includes('..')) {
    throw new ValidationError('File path cannot contain ".." (path traversal not allowed)', 'filePath');
  }

  // Block absolute paths unless explicitly allowed
  if (!allowAbsolutePaths && (filePath.startsWith('/') || /^[a-zA-Z]:/.test(filePath))) {
    throw new ValidationError('Absolute paths are not allowed', 'filePath');
  }

  // Block null bytes
  if (filePath.includes('\0')) {
    throw new ValidationError('File path cannot contain null bytes', 'filePath');
  }

  // Validate length
  if (filePath.length > 4096) {
    throw new ValidationError('File path exceeds maximum length of 4096 characters', 'filePath');
  }

  return true;
}

/**
 * Create a Zod schema that validates a non-empty string
 * @param fieldName - Name of the field for error messages
 * @returns Zod schema
 */
export function createNonEmptyStringSchema(fieldName: string = 'value') {
  return z.string()
    .min(1, `${fieldName} cannot be empty`)
    .refine(val => val.trim().length > 0, `${fieldName} cannot be only whitespace`);
}

/**
 * Create a Zod schema that validates an email
 * @returns Zod schema
 */
export function createEmailSchema() {
  return z.string()
    .email('Invalid email format')
    .max(254, 'Email cannot exceed 254 characters')
    .refine(val => EMAIL_REGEX.test(val), 'Invalid email format');
}

/**
 * Create a Zod schema that validates a URL
 * @param allowedProtocols - Array of allowed protocols
 * @returns Zod schema
 */
export function createUrlSchema(allowedProtocols: string[] = ['http', 'https']) {
  return z.string()
    .url('Invalid URL format')
    .refine(val => {
      try {
        const url = new URL(val);
        return allowedProtocols.includes(url.protocol.replace(':', ''));
      } catch {
        return false;
      }
    }, `URL protocol must be one of: ${allowedProtocols.join(', ')}`);
}

/**
 * Batch validate multiple values
 * @param validations - Array of validation functions to execute
 * @returns Object with validation results
 */
export function batchValidate(
  validations: Array<{ fn: () => boolean; field: string }>
): { isValid: boolean; errors: Array<{ field: string; message: string }> } {
  const errors: Array<{ field: string; message: string }> = [];

  for (const validation of validations) {
    try {
      validation.fn();
    } catch (error) {
      if (error instanceof ValidationError) {
        errors.push({
          field: validation.field,
          message: error.message
        });
      } else {
        errors.push({
          field: validation.field,
          message: 'Unknown validation error'
        });
      }
    }
  }

  return {
    isValid: errors.length === 0,
    errors
  };
}
