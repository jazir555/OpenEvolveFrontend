/**
 * useForm Hook
 * Enhanced form state management with validation
 */

import { useState, useCallback } from 'react';
import { validateWorkflowName, validateProblemStatement, validateEmail } from '../utils/validation';

export interface FormField {
  value: string;
  error: string | null;
  touched: boolean;
}

export interface FormState<T> {
  values: T;
  errors: Record<keyof T, string | null>;
  touched: Record<keyof T, boolean>;
  isValid: boolean;
  isDirty: boolean;
}

interface UseFormOptions<T> {
  initialValues: T;
  validate?: (values: T) => Record<keyof T, string | null>;
  onSubmit: (values: T) => void | Promise<void>;
}

export function useForm<T extends Record<string, unknown>>({
  initialValues,
  validate,
  onSubmit,
}: UseFormOptions<T>) {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Record<keyof T, string | null>>({} as Record<keyof T, string | null>);
  const [touched, setTouched] = useState<Record<keyof T, boolean>>({} as Record<keyof T, boolean>);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitCount, setSubmitCount] = useState(0);

  const setValue = useCallback(
    (name: keyof T, value: T[keyof T]) => {
      setValues((prev) => ({ ...prev, [name]: value }));
      setTouched((prev) => ({ ...prev, [name]: true }));

      // Validate field on change if it's been touched
      if (touched[name] && validate) {
        const validationErrors = validate({ ...values, [name]: value });
        setErrors((prev) => ({ ...prev, [name]: validationErrors[name] || null }));
      }
    },
    [values, touched, validate]
  );

  const setValuesBatch = useCallback((newValues: Partial<T>) => {
    setValues((prev) => ({ ...prev, ...newValues }));
    Object.keys(newValues).forEach((key) => {
      setTouched((prev) => ({ ...prev, [key as keyof T]: true }));
    });
  }, []);

  const setError = useCallback((name: keyof T, error: string | null) => {
    setErrors((prev) => ({ ...prev, [name]: error }));
  }, []);

  const setTouchedField = useCallback((name: keyof T) => {
    setTouched((prev) => ({ ...prev, [name]: true }));
  }, []);

  const validateForm = useCallback(() => {
    if (validate) {
      const validationErrors = validate(values);
      setErrors(validationErrors);
      setTouched(
        Object.keys(values).reduce(
          (acc, key) => ({ ...acc, [key]: true }),
          {} as Record<keyof T, boolean>
        )
      );
      return Object.values(validationErrors).every((error) => error === null);
    }
    return true;
  }, [values, validate]);

  const handleSubmit = useCallback(
    async (event?: React.FormEvent) => {
      if (event) {
        event.preventDefault();
      }

      setIsSubmitting(true);
      setSubmitCount((prev) => prev + 1);

      const isValid = validateForm();

      if (isValid) {
        try {
          await onSubmit(values);
        } catch (error) {
          console.error('Form submission error:', error);
        }
      }

      setIsSubmitting(false);
    },
    [values, validateForm, onSubmit]
  );

  const resetForm = useCallback(() => {
    setValues(initialValues);
    setErrors({} as Record<keyof T, string | null>);
    setTouched({} as Record<keyof T, boolean>);
    setSubmitCount(0);
  }, [initialValues]);

  const isValid = Object.values(errors).every((error) => error === null);
  const isDirty = Object.values(touched).some((t) => t);

  return {
    values,
    errors,
    touched,
    isValid,
    isDirty,
    isSubmitting,
    submitCount,
    setValue,
    setValuesBatch,
    setError,
    setTouched: setTouchedField,
    validate: validateForm,
    handleSubmit,
    resetForm,
  };
}

/**
 * useFormFields Hook
 * Simplified hook for managing multiple form fields
 */
export function useFormFields<T extends Record<string, string>>(initialValues: T) {
  const [fields, setFields] = useState<Record<keyof T, FormField>>(
    Object.keys(initialValues).reduce(
      (acc, key) => ({
        ...acc,
        [key]: {
          value: initialValues[key as keyof T],
          error: null,
          touched: false,
        },
      }),
      {} as Record<keyof T, FormField>
    )
  );

  const updateField = (name: keyof T, newValue: string) => {
    setFields((prev) => ({
      ...prev,
      [name]: {
        ...prev[name],
        value: newValue,
        touched: true,
      },
    }));
  };

  const setFieldError = (name: keyof T, error: string | null) => {
    setFields((prev) => ({
      ...prev,
      [name]: {
        ...prev[name],
        error,
      },
    }));
  };

  const validateField = (name: keyof T, validator: (value: string) => string | null) => {
    const field = fields[name];
    const error = validator(field.value);
    setFieldError(name, error);
    return error === null;
  };

  const getValues = () => {
    return Object.keys(fields).reduce(
      (acc, key) => ({
        ...acc,
        [key]: fields[key as keyof T].value,
      }),
      {} as T
    );
  };

  const isValid = Object.values(fields).every((field) => field.error === null);

  return {
    fields,
    updateField,
    setFieldError,
    validateField,
    getValues,
    isValid,
  };
}
