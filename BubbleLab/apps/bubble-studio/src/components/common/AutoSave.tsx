/**
 * AutoSave Component
 * Auto-save indicator with status display
 */

import { useState, useEffect } from 'react';
import { useLocalStorage } from '../../hooks/useLocalStorage';
import { formatDuration } from '../../utils/date';

interface AutoSaveProps {
  key: string;
  data: Record<string, unknown>;
  interval?: number;
  onSave?: () => void;
}

export function AutoSave({ key, data, interval = 30000, onSave }: AutoSaveProps) {
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const [savedData, setSavedData] = useLocalStorage(key, data);

  useEffect(() => {
    const autoSave = setInterval(() => {
      if (JSON.stringify(data) !== JSON.stringify(savedData)) {
        setIsSaving(true);
        setSaveStatus('saving');

        try {
          setSavedData(data);
          setLastSaved(new Date());
          setSaveStatus('saved');
          onSave?.();
        } catch (error) {
          setSaveStatus('error');
        } finally {
          setIsSaving(false);
          setTimeout(() => setSaveStatus('idle'), 2000);
        }
      }
    }, interval);

    return () => clearInterval(autoSave);
  }, [data, savedData, key, interval, setSavedData, onSave]);

  const statusConfig = {
    idle: { text: 'All changes saved', color: 'text-gray-500 dark:text-gray-400' },
    saving: { text: 'Saving...', color: 'text-blue-600 dark:text-blue-400' },
    saved: { text: 'Saved', color: 'text-green-600 dark:text-green-400' },
    error: { text: 'Failed to save', color: 'text-red-600 dark:text-red-400' },
  };

  const config = statusConfig[saveStatus];
  const hasUnsavedChanges = JSON.stringify(data) !== JSON.stringify(savedData);

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className={config.color}>{config.text}</span>

      {hasUnsavedChanges && (
        <span className="text-gray-500 dark:text-gray-400">
          (Unsaved changes)
        </span>
      )}

      {lastSaved && !hasUnsavedChanges && (
        <span className="text-gray-500 dark:text-gray-400">
          Last saved {lastSaved && <TimeAgo date={lastSaved} />}
        </span>
      )}

      {isSaving && (
        <svg className="animate-spin h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24">
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )}
    </div>
  );
}

function TimeAgo({ date }: { date: Date }) {
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return date.toLocaleDateString();
}
