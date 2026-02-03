/**
 * useTimeout Hook
 * Manage timeout
 */

import { useEffect, useRef, useCallback } from 'react';

export function useTimeout(callback: () => void, delay: number | null) {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const set = useCallback(() => {
    if (delay !== null) {
      timeoutRef.current = setTimeout(callback, delay);
    }
  }, [callback, delay]);

  const clear = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  }, []);

  const reset = useCallback(() => {
    clear();
    set();
  }, [clear, set]);

  useEffect(() => {
    set();
    return clear;
  }, [delay, set, clear]);

  return { reset, clear };
}

/**
 * useTimeoutFn Hook
 * Timeout with manual trigger
 */
export function useTimeoutFn() {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const set = useCallback((callback: () => void, delay: number) => {
    clear();
    timeoutRef.current = setTimeout(callback, delay);
  }, []);

  const clear = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  useEffect(() => {
    return clear;
  }, [clear]);

  return { set, clear };
}
