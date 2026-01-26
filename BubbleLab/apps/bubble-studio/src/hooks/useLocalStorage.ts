/**
 * useLocalStorage Hook
 * Sync state to localStorage with automatic persistence
 */

import { useState, useEffect, useCallback } from 'react';
import { storageGet, storageSet, storageRemove } from '../utils/storage';

export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void, () => void] {
  // Get stored value or use initial value
  const [storedValue, setStoredValue] = useState<T>(() => {
    return storageGet(key, initialValue);
  });

  // Update localStorage when value changes
  useEffect(() => {
    storageSet(key, storedValue);
  }, [key, storedValue]);

  // Return a wrapped version of useState's setter function
  const setValue = useCallback(
    (value: T | ((val: T) => T)) => {
      try {
        const valueToStore = value instanceof Function ? value(storedValue) : value;
        setStoredValue(valueToStore);
      } catch (error) {
        console.error(`Error setting localStorage key "${key}":`, error);
      }
    },
    [key, storedValue]
  );

  // Remove item from localStorage
  const removeValue = useCallback(() => {
    try {
      storageRemove(key);
      setStoredValue(initialValue);
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  }, [key, initialValue]);

  return [storedValue, setValue, removeValue];
}

/**
 * useLocalStorageForm Hook
 * Auto-save form drafts to localStorage
 */
export function useLocalStorageForm<T extends Record<string, unknown>>(
  key: string,
  initialValues: T
): [T, (values: Partial<T>) => void, () => void, boolean] {
  const [values, setValues, removeValues] = useLocalStorage<T>(key, initialValues);
  const [isDirty, setIsDirty] = useState(false);

  const updateValues = useCallback((newValues: Partial<T>) => {
    setValues((prev) => ({ ...prev, ...newValues }));
    setIsDirty(true);
  }, [setValues]);

  const clearForm = useCallback(() => {
    removeValues();
    setIsDirty(false);
  }, [removeValues]);

  return [values, updateValues, clearForm, isDirty];
}

/**
 * useLocalStorageArray Hook
 * Manage arrays in localStorage
 */
export function useLocalStorageArray<T>(
  key: string,
  initialValue: T[]
): [T[], (item: T) => void, (index: number) => void, () => void] {
  const [items, setItems] = useLocalStorage<T[]>(key, initialValue);

  const addItem = useCallback((item: T) => {
    setItems((prev) => [...prev, item]);
  }, [setItems]);

  const removeItem = useCallback((index: number) => {
    setItems((prev) => prev.filter((_, i) => i !== index));
  }, [setItems]);

  const clearItems = useCallback(() => {
    setItems([]);
  }, [setItems]);

  return [items, addItem, removeItem, clearItems];
}
