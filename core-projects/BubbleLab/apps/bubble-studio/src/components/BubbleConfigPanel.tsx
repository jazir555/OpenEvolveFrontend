import { useEffect, useState } from 'react';

/**
 * Reusable "Bubble Config" panel used by the OneKE and GKET pages.
 *
 * Persists a free-form configuration object to localStorage (keyed per bubble)
 * so the bubble can be fully configured before running. The shape is
 * intentionally generic (JSON) because each backend defines its own run
 * parameters; the persisted value can be read by the run flows later.
 */

interface BubbleConfigPanelProps {
  bubbleKey: string;
  /** Short hint shown above the editor. */
  hint?: string;
}

const storageKey = (bubbleKey: string) => `bubblelab.config.${bubbleKey}`;

export function BubbleConfigPanel({ bubbleKey, hint }: BubbleConfigPanelProps) {
  const [text, setText] = useState('{\n  \n}');
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const raw = localStorage.getItem(storageKey(bubbleKey));
    if (raw) setText(raw);
  }, [bubbleKey]);

  const save = () => {
    setError(null);
    try {
      JSON.parse(text);
    } catch (e) {
      setError(`Invalid JSON: ${(e as Error).message}`);
      return;
    }
    localStorage.setItem(storageKey(bubbleKey), text);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        {hint ??
          'Configure this bubble before running. Saved to local storage and reused by the run flow.'}
      </p>

      <label className="block text-sm font-medium text-gray-700">
        Configuration (JSON)
        <textarea
          className="mt-1 h-64 w-full rounded-md border border-gray-300 p-3 font-mono text-xs"
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
        />
      </label>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="flex items-center gap-3">
        <button
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white"
          onClick={save}
        >
          Save config
        </button>
        {saved && (
          <span className="text-sm text-green-700">Saved ✓</span>
        )}
      </div>
    </div>
  );
}
