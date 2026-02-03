/**
 * MultiSelect Component
 * Select multiple items from a list
 */

import { useState, useRef, useEffect } from 'react';
import { Tag } from './Tag';

interface MultiSelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

interface MultiSelectProps {
  options: MultiSelectOption[];
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

export function MultiSelect({
  options,
  value,
  onChange,
  placeholder = 'Select items...',
  disabled = false,
  className = '',
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter options based on search
  const filteredOptions = options.filter(
    (option) =>
      option.label.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !option.disabled
  );

  // Handle click outside to close dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleToggleOption = (optionValue: string) => {
    if (value.includes(optionValue)) {
      onChange(value.filter((v) => v !== optionValue));
    } else {
      onChange([...value, optionValue]);
    }
  };

  const handleRemove = (optionValue: string) => {
    onChange(value.filter((v) => v !== optionValue));
  };

  const selectedOptions = options.filter((opt) => value.includes(opt.value));

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      {/* Selected items display */}
      <div
        onClick={() => !disabled && setIsOpen(!isOpen)}
        className={`
          min-h-[38px] px-3 py-1.5 border rounded-md bg-white dark:bg-gray-800
          border-gray-300 dark:border-gray-600 cursor-pointer
          flex flex-wrap gap-1.5
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          ${isOpen ? 'ring-2 ring-blue-500 border-blue-500' : ''}
        `}
      >
        {selectedOptions.length === 0 ? (
          <span className="text-gray-500 dark:text-gray-400 text-sm">
            {placeholder}
          </span>
        ) : (
          selectedOptions.map((opt) => (
            <Tag key={opt.value} variant="blue" size="sm">
              {opt.label}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemove(opt.value);
                }}
                className="ml-1 text-gray-500 hover:text-gray-700 dark:text-gray-400"
              >
                ×
              </button>
            </Tag>
          ))
        )}
      </div>

      {/* Dropdown */}
      {isOpen && !disabled && (
        <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg max-h-60 overflow-auto">
          {/* Search input */}
          <div className="p-2 border-b border-gray-300 dark:border-gray-600">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search..."
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-700 dark:text-white"
              autoFocus
            />
          </div>

          {/* Options list */}
          <div className="py-1">
            {filteredOptions.length === 0 ? (
              <div className="px-3 py-2 text-sm text-gray-500 dark:text-gray-400">
                No options found
              </div>
            ) : (
              filteredOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => handleToggleOption(option.value)}
                  className={`
                    w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700
                    flex items-center gap-2
                  `}
                >
                  <input
                    type="checkbox"
                    checked={value.includes(option.value)}
                    onChange={() => {}}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="flex-1 dark:text-white">
                    {option.label}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
