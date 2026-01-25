/**
 * BubbleLab Compatible Select Component
 * 
 * A React select component that follows BubbleLab's design system
 */

import React from 'react';

interface BubbleSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
}

const BubbleSelect: React.FC<BubbleSelectProps> = ({
  label,
  error,
  className = '',
  children,
  ...props
}) => {
  const selectClasses = `flex h-10 w-full rounded-md border ${
    error 
      ? 'border-red-500 focus:border-red-500 focus:ring-red-500' 
      : 'border-gray-300 dark:border-gray-600 focus:border-blue-500 focus:ring-blue-500'
  } bg-white dark:bg-gray-700 px-3 py-2 text-sm text-gray-900 dark:text-white shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50 ${className}`;

  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          {label}
        </label>
      )}
      <select
        className={selectClasses}
        {...props}
      >
        {children}
      </select>
      {error && (
        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
};

export default BubbleSelect;