import { z } from 'zod';
import { errorLogger } from './errorLogging';

/**
 * Validation error interface
 */
export interface ValidationError {
  field: string;
  message: string;
  code?: string;
}

/**
 * Validation result interface
 */
export interface ValidationResult<T = any> {
  success: boolean;
  data?: T;
  errors?: ValidationError[];
  error?: string;
}

/**
 * Common validation schemas
 */
export const validationSchemas = {
  // Workflow validation
  workflow: z.object({
    name: z.string().min(1, 'Workflow name is required'),
    type: z.string().min(1, 'Workflow type is required'),
    config: z.record(z.unknown()),
  }).catch(() => ({
    name: 'Unnamed Workflow',
    type: 'standard',
    config: {},
  })),

  // Artifact validation
  artifact: z.object({
    type: z.string().min(1, 'Artifact type is required'),
    name: z.string().min(1, 'Artifact name is required'),
    content: z.unknown(),
    tags: z.array(z.string()).optional().default([]),
  }).catch(() => ({
    type: 'unknown',
    name: 'Unnamed Artifact',
    content: null,
    tags: [],
  })),

  // Lean proof validation
  leanProof: z.object({
    theorem: z.string().min(1, 'Theorem is required'),
    model: z.string().min(1, 'Model is required'),
  }).catch(() => ({
    theorem: '',
    model: '',
  })),

  // Knowledge search validation
  searchQuery: z.object({
    query: z.string().min(1, 'Search query is required'),
    type: z.string().optional(),
    tags: z.array(z.string()).optional().default([]),
    page: z.number().int().positive().optional().default(1),
    pageSize: z.number().int().positive().optional().default(10),
  }).catch(() => ({
    query: '',
    type: undefined,
    tags: [],
    page: 1,
    pageSize: 10,
  })),
};

/**
 * Validate data against a schema with comprehensive error handling
 */
export function validateData<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): ValidationResult<T> {
  try {
    const validated = schema.parse(data);
    return { success: true, data: validated };
  } catch (error) {
    if (error instanceof z.ZodError) {
      const errors: ValidationError[] = error.errors.map(err => ({
        field: err.path.join('.') || 'unknown',
        message: err.message,
        code: err.code,
      }));

      return {
        success: false,
        errors,
        error: errors[0]?.message || 'Validation failed'
      };
    }

    // Log unexpected validation errors
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { component: 'Validator', function: 'validateData', additionalData: { schema: schema._def.typeName } }
    );

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Validation failed due to unexpected error'
    };
  }
}

/**
 * Safe validation that never throws
 */
export function safeValidate<T>(
  schema: z.ZodSchema<T>,
  data: unknown,
  options: {
    logErrors?: boolean;
    fallbackValue?: T;
    fieldName?: string;
  } = {}
): ValidationResult<T> {
  const { logErrors = true, fallbackValue, fieldName = 'data' } = options;

  try {
    const result = validateData(schema, data);

    if (!result.success && logErrors) {
      const errorContext = {
        component: 'SafeValidator',
        function: 'safeValidate',
        additionalData: {
          fieldName,
          schemaType: schema._def.typeName,
          inputValue: typeof data === 'object' ? '[object]' : String(data)
        }
      };

      if (result.errors) {
        result.errors.forEach(validationError => {
          errorLogger.logError(
            new Error(`Validation failed for ${fieldName}: ${validationError.message}`),
            'warn',
            errorContext
          );
        });
      } else if (result.error) {
        errorLogger.logError(
          new Error(`Validation failed for ${fieldName}: ${result.error}`),
          'warn',
          errorContext
        );
      }
    }

    return result;
  } catch (error) {
    if (logErrors) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        {
          component: 'SafeValidator',
          function: 'safeValidate',
          additionalData: { fieldName, schemaType: schema._def.typeName }
        }
      );
    }

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unexpected validation error occurred',
      data: fallbackValue
    };
  }
}

/**
 * Validate multiple fields at once
 */
export function validateFields(
  validations: Array<{
    field: string;
    value: unknown;
    schema: z.ZodSchema<any>;
  }>
): ValidationResult<Record<string, any>> {
  try {
    const results: Record<string, any> = {};
    const errors: ValidationError[] = [];

    for (const validation of validations) {
      try {
        const fieldResult = validateData(validation.schema, validation.value);

        if (fieldResult.success) {
          results[validation.field] = fieldResult.data;
        } else {
          if (fieldResult.errors) {
            errors.push(...fieldResult.errors.map(err => ({
              ...err,
              field: `${validation.field}.${err.field}`.replace('..', '.')
            })));
          } else if (fieldResult.error) {
            errors.push({
              field: validation.field,
              message: fieldResult.error,
            });
          }
        }
      } catch (fieldError) {
        errors.push({
          field: validation.field,
          message: fieldError instanceof Error ? fieldError.message : 'Field validation failed',
        });

        errorLogger.logError(
          fieldError instanceof Error ? fieldError : new Error(String(fieldError)),
          'error',
          {
            component: 'Validator',
            function: 'validateFields',
            additionalData: { field: validation.field, value: validation.value }
          }
        );
      }
    }

    if (errors.length > 0) {
      return { success: false, errors };
    }

    return { success: true, data: results };
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { component: 'Validator', function: 'validateFields' }
    );

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unexpected error during field validation'
    };
  }
}

/**
 * Validate configuration object
 */
export function validateConfig<T extends Record<string, unknown>>(
  config: T,
  schema: z.ZodSchema<T>,
  options: { strict?: boolean } = {}
): ValidationResult<T> {
  const { strict = false } = options;

  try {
    // If strict mode, validate exactly against schema
    if (strict) {
      return validateData(schema, config);
    }

    // Otherwise, validate and merge with defaults
    const validated = schema.safeParse(config);

    if (validated.success) {
      return { success: true, data: validated.data };
    } else {
      const errors: ValidationError[] = validated.error.errors.map(err => ({
        field: err.path.join('.') || 'config',
        message: err.message,
        code: err.code,
      }));

      return { success: false, errors };
    }
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { component: 'Validator', function: 'validateConfig', additionalData: { config } }
    );

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Configuration validation failed'
    };
  }
}

/**
 * Create a validator function with error handling
 */
export function createValidator<T>(
  schema: z.ZodSchema<T>,
  options: {
    onError?: (error: ValidationError[], originalData: unknown) => void;
    onSuccess?: (data: T) => void;
    logErrors?: boolean;
  } = {}
): (data: unknown) => ValidationResult<T> {
  const { onError, onSuccess, logErrors = true } = options;

  return (data: unknown): ValidationResult<T> => {
    try {
      const result = validateData(schema, data);

      if (result.success) {
        onSuccess?.(result.data);
      } else {
        if (logErrors) {
          errorLogger.logError(
            new Error(`Validation failed: ${result.error || 'Multiple errors'}`),
            'warn',
            {
              component: 'Validator',
              function: 'createValidator',
              additionalData: {
                errors: result.errors,
                originalData: data
              }
            }
          );
        }
        onError?.(result.errors || [{ field: 'unknown', message: result.error || 'Validation failed' }], data);
      }

      return result;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'Validator', function: 'createValidator', additionalData: { data } }
      );

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unexpected error in validator'
      };
    }
  };
}

/**
 * Validate with transformation
 */
export function validateAndTransform<Input, Output>(
  data: unknown,
  transformer: (input: Input) => Output,
  inputSchema?: z.ZodSchema<Input>
): ValidationResult<Output> {
  try {
    // If input schema provided, validate first
    if (inputSchema) {
      const validationResult = validateData(inputSchema, data);

      if (!validationResult.success) {
        return validationResult as ValidationResult<Output>;
      }

      data = validationResult.data;
    }

    // Apply transformation
    const transformed = transformer(data as Input);

    return { success: true, data: transformed };
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      {
        component: 'Validator',
        function: 'validateAndTransform',
        additionalData: {
          originalData: data,
          hasInputSchema: !!inputSchema
        }
      }
    );

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Transformation failed'
    };
  }
}

/**
 * Batch validation utility
 */
export function batchValidate<T>(
  items: unknown[],
  schema: z.ZodSchema<T>,
  options: { continueOnError?: boolean } = {}
): Array<ValidationResult<T>> {
  const { continueOnError = true } = options;
  const results: Array<ValidationResult<T>> = [];

  for (let i = 0; i < items.length; i++) {
    try {
      const result = validateData(schema, items[i]);
      results.push(result);

      // If not continuing on error and this failed, stop processing
      if (!continueOnError && !result.success) {
        break;
      }
    } catch (error) {
      const validationError = error instanceof Error ? error : new Error(String(error));

      errorLogger.logError(
        validationError,
        'error',
        {
          component: 'Validator',
          function: 'batchValidate',
          additionalData: {
            itemIndex: i,
            itemValue: items[i]
          }
        }
      );

      results.push({
        success: false,
        error: validationError.message
      });

      // If not continuing on error, stop processing
      if (!continueOnError) {
        break;
      }
    }
  }

  return results;
}
