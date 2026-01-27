/**
 * RadioGroup Component
 * Single selection from multiple options
 */

import { RadioGroup as HeadlessRadioGroup } from '@headlessui/react';
import { ElementType } from 'react';

interface RadioOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

interface RadioGroupProps {
  name: string;
  label?: string;
  options: RadioOption[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  orientation?: 'horizontal' | 'vertical';
}

export function RadioGroup({
  name,
  label,
  options,
  value,
  onChange,
  disabled = false,
  orientation = 'vertical',
}: RadioGroupProps) {
  return (
    <HeadlessRadioGroup value={value} onChange={onChange} disabled={disabled}>
      {label && (
        <HeadlessRadioGroup.Label className="block text-sm font-medium text-gray-900 dark:text-white mb-2">
          {label}
        </HeadlessRadioGroup.Label>
      )}
      <div
        className={
          orientation === 'vertical'
            ? 'space-y-2'
            : 'flex gap-4 flex-wrap'
        }
      >
        {options.map((option) => (
          <HeadlessRadioGroup.Option
            key={option.value}
            value={option.value}
            disabled={option.disabled || disabled}
            className={({ checked, disabled: optDisabled }: any) => `
              relative flex items-start cursor-pointer
              ${optDisabled ? 'opacity-50 cursor-not-allowed' : ''}
              ${orientation === 'horizontal' ? 'flex-1 min-w-[200px]' : ''}
            `}
          >
            {({ checked }: { checked: boolean }) => (
              <>
                <div className="flex h-6 items-center">
                  <input
                    type="radio"
                    name={name}
                    checked={checked}
                    readOnly
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-600"
                  />
                </div>
                <div className="ml-3 text-sm">
                  <HeadlessRadioGroup.Label
                    as={'span' as ElementType}
                    className="block font-medium text-gray-900 dark:text-white"
                  >
                    {option.label}
                  </HeadlessRadioGroup.Label>
                  {option.description && (
                    <HeadlessRadioGroup.Description
                      as={'span' as ElementType}
                      className="text-gray-500 dark:text-gray-400"
                    >
                      {option.description}
                    </HeadlessRadioGroup.Description>
                  )}
                </div>
              </>
            )}
          </HeadlessRadioGroup.Option>
        ))}
      </div>
    </HeadlessRadioGroup>
  );
}
