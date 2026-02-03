/**
 * Checkbox Component
 * Reusable checkbox with label
 */

import { InputHTMLAttributes, forwardRef } from 'react';

interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  error?: string;
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ label, error, id, className = '', ...props }, ref) => {
    return (
      <div className="flex items-start">
        <div className="flex h-5 items-center">
          <input
            ref={ref}
            id={id}
            type="checkbox"
            className={`
              h-4 w-4 rounded border-gray-300 text-blue-600
              focus:ring-blue-500 dark:border-gray-600
              ${className}
            `}
            {...props}
          />
        </div>
        {label && (
          <label htmlFor={id} className="ml-2 text-sm text-gray-700 dark:text-gray-300">
            {label}
          </label>
        )}
        {error && (
          <p className="ml-6 text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
      </div>
    );
  }
);

Checkbox.displayName = 'Checkbox';
