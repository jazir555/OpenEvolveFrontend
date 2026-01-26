/**
 * useThrottle Hook
 * Throttle value changes
 */

import { useState, useEffect } from 'react';
import { throttle } from '../utils/debounce';

export function useThrottle<T>(value: T, delay: number): T {
  const [throttledValue, setThrottledValue] = useState<T>(value);

  useEffect(() => {
    const throttledUpdate = throttle((val: T) => {
      setThrottledValue(val);
    }, delay);

    throttledUpdate(value);

    return () => {
      throttledUpdate.cancel?.();
    };
  }, [value, delay]);

  return throttledValue;
}

/**
 * useThrottleCallback Hook
 * Throttle callback function
 */
export function useThrottleCallback<T extends (...args: never[]) => unknown>(
  callback: T,
  delay: number
): T {
  const [throttledCallback, setThrottledCallback] = useState<T>(() =>
    throttle((...args: never[]) => callback(...args), delay) as unknown as T
  );

  useEffect(() => {
    setThrottledCallback(
      throttle((...args: never[]) => callback(...args), delay) as unknown as T
    );
  }, [callback, delay]);

  return throttledCallback;
}
