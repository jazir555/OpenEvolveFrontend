/**
 * Form Component
 * Reusable form with validation
 */

import { ReactNode } from 'react';
import { Button } from './Button';

interface FormProps {
  onSubmit: (e: React.FormEvent) => void;
  children: ReactNode;
  isLoading?: boolean;
  submitLabel?: string;
  cancelLabel?: string;
  onCancel?: () => void;
}

export function Form({ onSubmit, children, isLoading, submitLabel = 'Submit', cancelLabel, onCancel }: FormProps) {
  return (
    <form onSubmit={onSubmit} className="space-y-4">
      {children}
      <div className="flex items-center justify-end gap-2">
        {onCancel && cancelLabel && (
          <Button
            variant="ghost"
            onClick={onCancel}
            type="button"
          >
            {cancelLabel}
          </Button>
        )}
        <Button type="submit" isLoading={isLoading}>
          {submitLabel}
        </Button>
      </div>
    </form>
  );
}
