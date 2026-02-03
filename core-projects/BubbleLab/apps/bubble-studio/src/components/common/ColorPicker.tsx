/**
 * ColorPicker Component
 * Color selection input
 */

import { useState } from 'react';

interface ColorPickerProps {
  value: string;
  onChange: (color: string) => void;
  label?: string;
  presetColors?: string[];
  className?: string;
}

const DEFAULT_PRESETS = [
  '#ef4444', // red-500
  '#f97316', // orange-500
  '#eab308', // yellow-500
  '#22c55e', // green-500
  '#06b6d4', // cyan-500
  '#3b82f6', // blue-500
  '#8b5cf6', // violet-500
  '#ec4899', // pink-500
  '#6b7280', // gray-500
  '#000000', // black
  '#ffffff', // white
];

export function ColorPicker({
  value,
  onChange,
  label,
  presetColors = DEFAULT_PRESETS,
  className = '',
}: ColorPickerProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className={`relative ${className}`}>
      {label && (
        <label className="block text-sm font-medium text-gray-900 dark:text-white mb-1">
          {label}
        </label>
      )}

      <div className="flex gap-2">
        {/* Color preview and input */}
        <div className="relative">
          <input
            type="color"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-10 h-10 rounded cursor-pointer border-2 border-gray-300 dark:border-gray-600"
          />
          <div
            className="absolute inset-0 rounded pointer-events-none border-2 border-white dark:border-gray-800"
            style={{ backgroundColor: value }}
          />
        </div>

        {/* Hex input */}
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white"
          placeholder="#000000"
        />

        {/* Preset toggle */}
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 dark:text-white"
        >
          {isOpen ? '▲' : '▼'}
        </button>
      </div>

      {/* Preset colors */}
      {isOpen && (
        <div className="mt-2 p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800">
          <div className="grid grid-cols-11 gap-1">
            {presetColors.map((color) => (
              <button
                key={color}
                type="button"
                onClick={() => {
                  onChange(color);
                  setIsOpen(false);
                }}
                className="w-8 h-8 rounded border-2 border-gray-300 dark:border-gray-600 hover:scale-110 transition-transform"
                style={{ backgroundColor: color }}
                title={color}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
