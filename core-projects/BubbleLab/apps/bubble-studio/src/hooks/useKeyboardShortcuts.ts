/**
 * useKeyboardShortcuts Hook
 * Register and handle keyboard shortcuts
 */

import { useEffect, useRef } from 'react';

interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;
  handler: (event: KeyboardEvent) => void;
  description?: string;
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[], enabled = true) {
  const handlersRef = useRef<Map<string, KeyboardShortcut['handler']>>(new Map());

  useEffect(() => {
    if (!enabled) return;

    // Build key map
    const keyMap = new Map<string, KeyboardShortcut['handler']>();

    shortcuts.forEach((shortcut) => {
      const keyCombo = buildKeyCombo(shortcut);
      keyMap.set(keyCombo, shortcut.handler);
    });

    handlersRef.current = keyMap;

    // Handle keydown events
    const handleKeyDown = (event: KeyboardEvent) => {
      const keyCombo = buildKeyCombo({
        key: event.key,
        ctrlKey: event.ctrlKey,
        shiftKey: event.shiftKey,
        altKey: event.altKey,
        metaKey: event.metaKey,
      });

      const handler = keyMap.get(keyCombo);
      if (handler) {
        event.preventDefault();
        handler(event);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, enabled]);

  return handlersRef.current;
}

function buildKeyCombo(shortcut: {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;
}): string {
  const parts: string[] = [];

  if (shortcut.ctrlKey) parts.push('ctrl');
  if (shortcut.shiftKey) parts.push('shift');
  if (shortcut.altKey) parts.push('alt');
  if (shortcut.metaKey) parts.push('meta');

  parts.push(shortcut.key.toLowerCase());

  return parts.join('+');
}

/**
 * useKeyboardShortcut Hook
 * Register a single keyboard shortcut
 */
export function useKeyboardShortcut(
  key: string,
  handler: (event: KeyboardEvent) => void,
  options: {
    ctrlKey?: boolean;
    shiftKey?: boolean;
    altKey?: boolean;
    metaKey?: boolean;
    enabled?: boolean;
  } = {}
) {
  const {
    ctrlKey = false,
    shiftKey = false,
    altKey = false,
    metaKey = false,
    enabled = true,
  } = options;

  useKeyboardShortcuts(
    [
      {
        key,
        ctrlKey,
        shiftKey,
        altKey,
        metaKey,
        handler,
      },
    ],
    enabled
  );
}
