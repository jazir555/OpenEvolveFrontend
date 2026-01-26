/**
 * Validation Utilities
 * Common validation functions for forms
 */

export interface ValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
}

export function validateWorkflowName(name: string): ValidationResult {
  const errors: Record<string, string> = {};

  if (!name || name.trim().length === 0) {
    errors.name = 'Workflow name is required';
  } else if (name.length < 3) {
    errors.name = 'Workflow name must be at least 3 characters';
  } else if (name.length > 100) {
    errors.name = 'Workflow name must not exceed 100 characters';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}

export function validateProblemStatement(statement: string): ValidationResult {
  const errors: Record<string, string> = {};

  if (!statement || statement.trim().length === 0) {
    errors.problem_statement = 'Problem statement is required';
  } else if (statement.length < 10) {
    errors.problem_statement = 'Problem statement must be at least 10 characters';
  } else if (statement.length > 10000) {
    errors.problem_statement = 'Problem statement must not exceed 10,000 characters';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}

export function validateApiKey(apiKey: string): ValidationResult {
  const errors: Record<string, string> = {};

  if (!apiKey || apiKey.trim().length === 0) {
    errors.api_key = 'API key is required';
  } else if (apiKey.length < 20) {
    errors.api_key = 'API key appears to be invalid';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}

export function validateEmail(email: string): ValidationResult {
  const errors: Record<string, string> = {};
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  if (!email || email.trim().length === 0) {
    errors.email = 'Email is required';
  } else if (!emailRegex.test(email)) {
    errors.email = 'Invalid email format';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}

export function validateNumber(value: string, min: number, max: number): ValidationResult {
  const errors: Record<string, string> = {};

  if (!value || value.trim().length === 0) {
    errors.value = 'Value is required';
  } else {
    const num = parseFloat(value);
    if (isNaN(num)) {
      errors.value = 'Must be a valid number';
    } else if (num < min || num > max) {
      errors.value = `Must be between ${min} and ${max}`;
    }
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}
