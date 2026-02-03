/**
 * Input Validator
 * Provides input validation and sanitization
 */

export class ValidationError extends Error {
  constructor(
    message: string,
    public readonly errors: string[]
  ) {
    super(message);
    this.name = 'ValidationError';
  }
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
export function sanitizeString(input: string): string {
  if (typeof input !== 'string') {
    return String(input);
  }

  return input
    .trim()
    .replace(/[<>]/g, '') // Remove potential HTML tags
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, ''); // Remove control characters
}

/**
 * Validate and sanitize input against a schema
 * @param input Raw input
 * @param schema Validation schema
 * @returns ValidationResult
 */
export function validateInput<T = any>(
  input: any,
  schema: ValidationSchema | ObjectSchema
): ValidationResult<T> {
  const errors: string[] = [];

  // Object validation
  if (schema.type === 'object') {
    if (typeof input !== 'object' || input === null || Array.isArray(input)) {
      return {
        success: false,
        errors: ['Input must be an object'],
      };
    }

    const objSchema = schema as ObjectSchema;
    const validatedData: any = {};

    for (const [key, fieldSchema] of Object.entries(objSchema)) {
      const value = input[key];

      // Check required fields
      if (fieldSchema.required && (value === undefined || value === null)) {
        errors.push(`Field '${key}' is required`);
        continue;
      }

      // Skip validation if not required and value is undefined
      if (!fieldSchema.required && value === undefined) {
        continue;
      }

      // Validate field
      const fieldResult = validateField(value, fieldSchema, key);
      if (!fieldResult.success) {
        errors.push(...(fieldResult.errors || []));
      } else {
        validatedData[key] = fieldResult.data;
      }
    }

    if (errors.length > 0) {
      return { success: false, errors };
    }

    return { success: true, data: validatedData as T };
  }

  // Primitive type validation
  const result = validateField(input, schema as ValidationSchema);
  if (!result.success) {
    return { success: false, errors: result.errors };
  }

  return { success: true, data: result.data as T };
}

/**
 * Validate a single field
 * @param value Field value
 * @param schema Field schema
 * @param fieldName Field name for error messages
 * @returns ValidationResult
 */
function validateField(
  value: any,
  schema: ValidationSchema,
  fieldName: string = 'field'
): ValidationResult {
  const errors: string[] = [];

  // Type validation
  if (schema.type === 'string') {
    if (typeof value !== 'string') {
      return {
        success: false,
        errors: [`${fieldName} must be a string`],
      };
    }

    let sanitizedValue = value;

    if (schema.sanitize) {
      sanitizedValue = sanitizeString(value);
    }

    // Length validation
    if (schema.minLength !== undefined && sanitizedValue.length < schema.minLength) {
      errors.push(`${fieldName} must be at least ${schema.minLength} characters`);
    }

    if (schema.maxLength !== undefined && sanitizedValue.length > schema.maxLength) {
      errors.push(`${fieldName} must be at most ${schema.maxLength} characters`);
    }

    // Pattern validation
    if (schema.pattern && !schema.pattern.test(sanitizedValue)) {
      errors.push(`${fieldName} does not match required pattern`);
    }

    if (errors.length > 0) {
      return { success: false, errors };
    }

    return { success: true, data: sanitizedValue };
  }

  if (schema.type === 'number') {
    if (typeof value !== 'number' || isNaN(value)) {
      return {
        success: false,
        errors: [`${fieldName} must be a number`],
      };
    }

    if (schema.min !== undefined && value < schema.min) {
      errors.push(`${fieldName} must be at least ${schema.min}`);
    }

    if (schema.max !== undefined && value > schema.max) {
      errors.push(`${fieldName} must be at most ${schema.max}`);
    }

    if (errors.length > 0) {
      return { success: false, errors };
    }

    return { success: true, data: value };
  }

  if (schema.type === 'boolean') {
    if (typeof value !== 'boolean') {
      return {
        success: false,
        errors: [`${fieldName} must be a boolean`],
      };
    }
    return { success: true, data: value };
  }

  if (schema.type === 'array') {
    if (!Array.isArray(value)) {
      return {
        success: false,
        errors: [`${fieldName} must be an array`],
      };
    }

    if (schema.minLength !== undefined && value.length < schema.minLength) {
      errors.push(`${fieldName} must have at least ${schema.minLength} items`);
    }

    if (schema.maxLength !== undefined && value.length > schema.maxLength) {
      errors.push(`${fieldName} must have at most ${schema.maxLength} items`);
    }

    if (errors.length > 0) {
      return { success: false, errors };
    }

    return { success: true, data: value };
  }

  return {
    success: false,
    errors: [`Unsupported type: ${schema.type}`],
  };
}

/**
 * Validate that input is safe from code injection
 * @param input Input to check
 * @returns true if safe, false otherwise
 */
export function isSafeFromInjection(input: string): boolean {
  // Check for dangerous patterns
  const dangerousPatterns = [
    /eval\s*\(/i,
    /Function\s*\(/i,
    /setTimeout\s*\(\s*['"]/i,
    /setInterval\s*\(\s*['"]/i,
    /new\s+Function/i,
    /require\s*\(/i,
    /import\s*\(/i,
    /__proto__/i,
    /constructor\s*\[/i,
    /\.call\s*\(/i,
    /\.apply\s*\(/i,
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(input)) {
      return false;
    }
  }

  return true;
}

/**
 * Validate and throw if invalid
 * @param input Input to validate
 * @param schema Validation schema
 * @throws ValidationError if validation fails
 */
export function validateOrThrow<T = any>(
  input: any,
  schema: ValidationSchema | ObjectSchema
): T {
  const result = validateInput<T>(input, schema);

  if (!result.success) {
    throw new ValidationError(
      'Validation failed',
      result.errors || []
    );
  }

  return result.data!;
}
