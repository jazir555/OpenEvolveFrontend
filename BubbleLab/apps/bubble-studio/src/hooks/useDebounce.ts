/**
 * useDebounce Hook
 * Debounce value changes
 */

import { useState, useEffect } from 'react';
import { debounce } from '../utils/debounce';

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const debouncedUpdate = debounce((val: T) => {
      setDebouncedValue(val);
    }, delay);

    debouncedUpdate(value);

    return () => {
      // Cancel pending debounced calls
      (debouncedUpdate as any).cancel?.();
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * useDebounceCallback Hook
 * Debounce callback function
 */
export function useDebounceCallback<T extends (...args: any[]) => unknown>(
  callback: T,
  delay: number
): T {
  const [debouncedCallback, setDebouncedCallback] = useState<T>(() =>
    debounce((...args: any[]) => callback(...args), delay) as unknown as T
  );

  useEffect(() => {
    setDebouncedCallback(
      debounce((...args: any[]) => callback(...args), delay) as unknown as T
    );
  }, [callback, delay]);

  return debouncedCallback;
}
