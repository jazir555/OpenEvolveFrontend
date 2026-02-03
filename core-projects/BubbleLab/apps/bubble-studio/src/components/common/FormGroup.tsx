/**
 * FormGroup Component
 * Group related form fields together
 */

import { ReactNode } from 'react';

interface FormGroupProps {
  title?: string;
  description?: string;
  children: ReactNode;
  className?: string;
}

export function FormGroup({
  title,
  description,
  children,
  className = '',
}: FormGroupProps) {
  return (
    <div className={`border border-gray-200 dark:border-gray-700 rounded-lg p-4 ${className}`}>
      {(title || description) && (
        <div className="mb-4">
          {title && (
            <h3 className="text-base font-semibold text-gray-900 dark:text-white">
              {title}
            </h3>
          )}
          {description && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {description}
            </p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
