/**
 * Utility functions for OpenEvolve Integration Library
 */

import type {
  ValidationResult,
  ParameterSchema,
  ValidationErrorItem
} from '../api/types';

/**
 * Validate inputs against a JSON schema
 */
export function validateInputs(
  inputs: any,
  schema: ParameterSchema
): ValidationResult {
  const errors: ValidationErrorItem[] = [];
  const warnings: ValidationErrorItem[] = [];

  if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) {
    return {
      valid: false,
      errors: [{
        field: 'root',
        message: 'Inputs must be an object',
        code: 'INVALID_INPUT_TYPE'
      }],
      warnings: []
    };
  }

  // Check required fields
  if (schema.required) {
    for (const field of schema.required) {
      if (!(field in inputs) || inputs[field] === undefined) {
        errors.push({
          field,
          message: `Required field '${field}' is missing`,
          code: 'REQUIRED_FIELD_MISSING'
        });
      }
    }
  }

  // Check each property
  if (schema.properties) {
    for (const [fieldName, value] of Object.entries(inputs)) {
      const property = schema.properties[fieldName];
      if (!property) {
        warnings.push({
          field: fieldName,
          message: `Unknown field '${fieldName}'`,
          code: 'UNKNOWN_FIELD'
        });
        continue;
      }

      // Type validation
      if (value !== null && value !== undefined) {
        const typeError = validateType(fieldName, value, property);
        if (typeError) {
          errors.push(typeError);
        }

        // Enum validation
        if (property.enum && !property.enum.includes(value)) {
          errors.push({
            field: fieldName,
            message: `Value must be one of: ${property.enum.join(', ')}`,
            code: 'INVALID_ENUM_VALUE'
          });
        }

        // Range validation
        if (typeof value === 'number') {
          if (property.minimum !== undefined && value < property.minimum) {
            errors.push({
              field: fieldName,
              message: `Value must be at least ${property.minimum}`,
              code: 'VALUE_TOO_SMALL'
            });
          }
          if (property.maximum !== undefined && value > property.maximum) {
            errors.push({
              field: fieldName,
              message: `Value must be at most ${property.maximum}`,
              code: 'VALUE_TOO_LARGE'
            });
          }
        }

        // Pattern validation
        if (typeof value === 'string' && property.pattern) {
          try {
            const regex = new RegExp(property.pattern);
            if (!regex.test(value)) {
              errors.push({
                field: fieldName,
                message: `Value does not match required pattern`,
                code: 'PATTERN_MISMATCH'
              });
            }
          } catch (e) {
            warnings.push({
              field: fieldName,
              message: `Invalid regex pattern: ${property.pattern}`,
              code: 'INVALID_PATTERN'
            });
          }
        }
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Validate a value against a type
 */
function validateType(
  fieldName: string,
  value: any,
  property: any
): ValidationErrorItem | null {
  const expectedType = property.type;

  if (value === null) {
    if (expectedType === 'null') return null;
    return {
      field: fieldName,
      message: `Expected ${expectedType}, got null`,
      code: 'TYPE_MISMATCH'
    };
  }

  // Handle array types
  if (expectedType === 'array') {
    if (!Array.isArray(value)) {
      return {
        field: fieldName,
        message: `Expected array, got ${typeof value}`,
        code: 'TYPE_MISMATCH'
      };
    }
  }

  // Handle object types
  if (expectedType === 'object') {
    if (typeof value !== 'object' || Array.isArray(value)) {
      return {
        field: fieldName,
        message: `Expected object, got ${typeof value}`,
        code: 'TYPE_MISMATCH'
      };
    }
  }

  // Handle primitive types
  if (expectedType === 'string' && typeof value !== 'string') {
    return {
      field: fieldName,
      message: `Expected string, got ${typeof value}`,
      code: 'TYPE_MISMATCH'
    };
  }

  if (expectedType === 'number' && (typeof value !== 'number' || isNaN(value))) {
    return {
      field: fieldName,
      message: `Expected number, got ${typeof value}`,
      code: 'TYPE_MISMATCH'
    };
  }

  if (expectedType === 'integer') {
    if (!Number.isInteger(value)) {
      return {
        field: fieldName,
        message: `Expected integer, got ${typeof value === 'number' && isNaN(value) ? 'NaN' : typeof value}`,
        code: 'TYPE_MISMATCH'
      };
    }
  }

  if (expectedType === 'boolean' && typeof value !== 'boolean') {
    return {
      field: fieldName,
      message: `Expected boolean, got ${typeof value}`,
      code: 'TYPE_MISMATCH'
    };
  }

  return null;
}

/**
 * Deep merge two objects
 */
export function deepMerge<T extends object>(target: T, source: Partial<T>): T {
  const result = { ...target };

  for (const key in source) {
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
      continue;
    }
    
    const sourceValue = source[key];
    if (sourceValue === undefined) {
      continue;
    }

    const targetValue = (target as any)[key];

    if (isPlainObject(sourceValue) && isPlainObject(targetValue)) {
      (result as any)[key] = deepMerge(targetValue as any, sourceValue as any);
    } else if (Array.isArray(sourceValue)) {
      (result as any)[key] = [...sourceValue];
    } else {
      (result as any)[key] = sourceValue;
    }
  }

  return result;
}

/**
 * Generate a unique ID
 */
export function generateId(): string {
  // Use a more robust random ID generation
  return Math.random().toString(36).substring(2, 15) + 
         Math.random().toString(36).substring(2, 15);
}

/**
 * Format a duration in milliseconds to human-readable string
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

/**
 * Retry a function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000,
  shouldRetry?: (error: any) => boolean,
  onRetry?: (error: any, attempt: number, delay: number) => void
): Promise<T> {
  let lastError: any;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;

      if (attempt < maxRetries && (!shouldRetry || shouldRetry(error))) {
        let delay = baseDelay * Math.pow(2, attempt);
        
        // Respect RateLimitError retryAfter if available
        if (error.name === 'RateLimitError' && error.getRetryAfterMs) {
          delay = Math.max(delay, error.getRetryAfterMs());
        } else if (error.details?.retryAfter) {
          delay = Math.max(delay, error.details.retryAfter * 1000);
        }

        // Add jitter (±20%)
        const jitter = delay * 0.2 * (Math.random() * 2 - 1);
        delay = Math.max(0, delay + jitter);

        if (onRetry) {
          onRetry(error, attempt + 1, delay);
        }

        await sleep(delay);
      } else {
        throw error;
      }
    }
  }

  throw lastError!;
}

/**
 * Sleep for a specified duration
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Debounce a function
 */
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout;

  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Throttle a function
 */
export function throttle<T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let lastArgs: Parameters<T> | null = null;
  let inThrottle: boolean = false;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
        if (lastArgs) {
          fn(...lastArgs);
          lastArgs = null;
        }
      }, limit);
    } else {
      lastArgs = args;
    }
  };
}

/**
 * Parse a duration string to milliseconds
 */
export function parseDuration(duration: string): number {
  const match = duration.match(/^(\d+)(ms|s|m|h)$/);
  if (!match) {
    throw new Error(`Invalid duration format: ${duration}`);
  }

  const value = parseInt(match[1], 10);
  const unit = match[2];

  switch (unit) {
    case 'ms':
      return value;
    case 's':
      return value * 1000;
    case 'm':
      return value * 60 * 1000;
    case 'h':
      return value * 60 * 60 * 1000;
    default:
      throw new Error(`Invalid duration unit: ${unit}`);
  }
}

/**
 * Check if a value is a plain object
 */
export function isPlainObject(value: any): boolean {
  if (value === null || typeof value !== 'object') {
    return false;
  }
  const proto = Object.getPrototypeOf(value);
  return proto === null || proto === Object.prototype;
}

/**
 * Clone an object deeply
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as any;
  }

  if (obj instanceof RegExp) {
    return new RegExp(obj.source, obj.flags) as any;
  }

  if (obj instanceof Set) {
    const result = new Set();
    obj.forEach(value => result.add(deepClone(value)));
    return result as any;
  }

  if (obj instanceof Map) {
    const result = new Map();
    obj.forEach((value, key) => result.set(key, deepClone(value)));
    return result as any;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => deepClone(item)) as any;
  }

  const cloned: any = {};
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      cloned[key] = deepClone((obj as any)[key]);
    }
  }

  return cloned;
}

/**
 * Pick specific keys from an object
 */
export function pick<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    if (key in obj) {
      result[key] = obj[key];
    }
  }
  return result;
}

/**
 * Omit specific keys from an object
 */
export function omit<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj };
  for (const key of keys) {
    delete result[key];
  }
  return result as Omit<T, K>;
}
