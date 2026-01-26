/**
 * Slider Component
 * Range slider with value display
 */

import { InputHTMLAttributes, forwardRef, useState } from 'react';

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  helperText?: string;
  showValue?: boolean;
  min?: number;
  max?: number;
  step?: number;
}

export const Slider = forwardRef<HTMLInputElement, SliderProps>(
  ({ label, helperText, showValue = true, min = 0, max = 100, step = 1, value, ...props }, ref) => {
    const displayValue = value !== undefined ? value : parseFloat(props.defaultValue as string) || min;

    return (
      <div>
        {label && (
          <div className="mb-1 flex items-center justify-between">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              {label}
            </label>
            {showValue && (
              <span className="text-sm text-gray-900 dark:text-white">
                {typeof displayValue === 'number' ? displayValue.toFixed(2) : displayValue}
              </span>
            )}
          </div>
        )}
        <input
          ref={ref}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          className="block w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 dark:bg-gray-700 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-600 [&::-webkit-slider-thumb]:cursor-pointer"
          {...props}
        />
        {helperText && (
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{helperText}</p>
        )}
      </div>
    );
  }
);

Slider.displayName = 'Slider';
