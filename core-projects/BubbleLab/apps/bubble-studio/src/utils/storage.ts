/**
 * Storage Utilities
 * LocalStorage helpers with type safety
 */

export function storageGet<T>(key: string, defaultValue: T): T {
  if (typeof window === 'undefined') return defaultValue;

  try {
    const item = window.localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.error(`Error reading from localStorage key "${key}":`, error);
    return defaultValue;
  }
}

export function storageSet<T>(key: string, value: T): boolean {
  if (typeof window === 'undefined') return false;

  try {
    window.localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error(`Error writing to localStorage key "${key}":`, error);
    return false;
  }
}

export function storageRemove(key: string): boolean {
  if (typeof window === 'undefined') return false;

  try {
    window.localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error(`Error removing localStorage key "${key}":`, error);
    return false;
  }
}

export function storageClear(): boolean {
  if (typeof window === 'undefined') return false;

  try {
    window.localStorage.clear();
    return true;
  } catch (error) {
    console.error('Error clearing localStorage:', error);
    return false;
  }
}

// Typed storage helpers
export const STORAGE_KEYS = {
  LLM_CONFIG: 'openevolve_llm_config',
  UI_PREFERENCES: 'openevolve_ui_preferences',
  WORKFLOW_DRAFTS: 'openevolve_workflow_drafts',
  RECENT_WORKFLOWS: 'openevolve_recent_workflows',
} as const;
