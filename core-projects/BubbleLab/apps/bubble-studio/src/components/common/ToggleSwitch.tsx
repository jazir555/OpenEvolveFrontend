/**
 * ToggleSwitch Component
 * On/off toggle switch
 */

import { Switch } from '@headlessui/react';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
}

export function ToggleSwitch({ checked, onChange, label, description, disabled = false }: ToggleProps) {
  return (
    <div className="flex items-center justify-between">
      <span className="flex-grow">
        {label && (
          <span className="text-sm font-medium text-gray-900 dark:text-white">{label}</span>
        )}
        {description && (
          <p className="text-sm text-gray-500 dark:text-gray-400">{description}</p>
        )}
      </span>
      <Switch
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        className={`
          relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent
          transition-colors duration-200 ease-in-out
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
          ${
            checked
              ? 'bg-blue-600'
              : 'bg-gray-200 dark:bg-gray-700'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <span
          className={`
            pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow
            ring-0 transition duration-200 ease-in-out
            ${checked ? 'translate-x-5' : 'translate-x-0'}
          `}
        />
      </Switch>
    </div>
  );
}
