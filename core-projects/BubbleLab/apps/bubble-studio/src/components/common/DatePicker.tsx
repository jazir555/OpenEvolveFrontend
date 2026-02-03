/**
 * DatePicker Component
 * Date selection input
 */

import { useState } from 'react';

interface DatePickerProps {
  value?: string;
  onChange: (date: string) => void;
  label?: string;
  placeholder?: string;
  min?: string;
  max?: string;
  disabled?: boolean;
  className?: string;
}

export function DatePicker({
  value,
  onChange,
  label,
  placeholder = 'Select date...',
  min,
  max,
  disabled = false,
  className = '',
}: DatePickerProps) {
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const date = e.target.value;

    // Validate date format
    if (date && !isValidDate(date)) {
      setError('Invalid date format');
      return;
    }

    // Validate min/max
    if (min && date < min) {
      setError(`Date must be after ${min}`);
      return;
    }

    if (max && date > max) {
      setError(`Date must be before ${max}`);
      return;
    }

    setError(null);
    onChange(date);
  };

  return (
    <div className={className}>
      {label && (
        <label className="block text-sm font-medium text-gray-900 dark:text-white mb-1">
          {label}
        </label>
      )}

      <input
        type="date"
        value={value || ''}
        onChange={handleChange}
        min={min}
        max={max}
        disabled={disabled}
        className={`
          w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:text-white
          ${
            error
              ? 'border-red-500 focus:ring-red-500'
              : 'border-gray-300 dark:border-gray-600 focus:ring-blue-500'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      />

      {error && (
        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}

function isValidDate(dateString: string): boolean {
  const date = new Date(dateString);
  return date instanceof Date && !isNaN(date.getTime());
}
