/**
 * usePrevious Hook
 * Get the previous value of a state or prop
 */

import { useRef, useEffect } from 'react';

export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>(undefined);

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}

/**
 * usePreviousDistinct Hook
 * Get previous value only when it changes
 */
export function usePreviousDistinct<T>(value: T): T | undefined {
  const prevRef = useRef<T>(undefined);
  const currentRef = useRef<T>(undefined);

  if (currentRef.current !== value) {
    prevRef.current = currentRef.current;
    currentRef.current = value;
  }

  return prevRef.current;
}
