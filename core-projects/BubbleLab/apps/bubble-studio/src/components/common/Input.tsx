/**
 * Input Component
 * Reusable text input with label and error handling
 */

import { InputHTMLAttributes, forwardRef } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, helperText, className = '', ...props }, ref) => {
    const inputStyles = `
      block w-full rounded-lg border px-3 py-2 text-sm
      focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500
      disabled:bg-gray-100 disabled:cursor-not-allowed
      ${error ? 'border-red-500' : 'border-gray-300'}
      dark:border-gray-600 dark:bg-gray-700 dark:text-white
      ${className}
    `;

    return (
      <div>
        {label && (
          <label className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </label>
        )}
        <input ref={ref} className={inputStyles} {...props} />
        {error && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
        {helperText && !error && (
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{helperText}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';
