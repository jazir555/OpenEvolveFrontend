/**
 * Prompt Component
 * User prompt with confirmation
 */

import { Modal } from './Modal';

interface PromptProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'danger' | 'warning' | 'info';
  onConfirm: () => void;
  onCancel: () => void;
}

export function Prompt({
  isOpen,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'info',
  onConfirm,
  onCancel,
}: PromptProps) {
  const variantStyles = {
    danger: {
      confirm: 'bg-red-600 hover:bg-red-700 focus:ring-red-500',
      icon: '⚠️',
    },
    warning: {
      confirm: 'bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500',
      icon: '⚠️',
    },
    info: {
      confirm: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500',
      icon: 'ℹ️',
    },
  };

  const styles = variantStyles[variant];

  return (
    <Modal isOpen={isOpen} onClose={onCancel} size="sm">
      <div className="text-center">
        <div className="text-4xl mb-4">{styles.icon}</div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          {title}
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
          {message}
        </p>

        <div className="flex justify-center gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className={`px-4 py-2 text-sm font-medium text-white rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 ${styles.confirm}`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </Modal>
  );
}
