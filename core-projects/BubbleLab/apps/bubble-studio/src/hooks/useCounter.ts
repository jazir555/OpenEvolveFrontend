/**
 * useCounter Hook
 * Manage counter state
 */

import { useCallback, useState } from 'react';

interface UseCounterOptions {
  min?: number;
  max?: number;
  step?: number;
  initial?: number;
}

export function useCounter(options: UseCounterOptions = {}) {
  const {
    min = -Infinity,
    max = Infinity,
    step = 1,
    initial = 0,
  } = options;

  const [count, setCount] = useState(initial);

  const increment = useCallback(() => {
    setCount((prev) => Math.min(prev + step, max));
  }, [max, step]);

  const decrement = useCallback(() => {
    setCount((prev) => Math.max(prev - step, min));
  }, [min, step]);

  const reset = useCallback(() => {
    setCount(initial);
  }, [initial]);

  const set = useCallback((value: number) => {
    setCount(Math.max(min, Math.min(max, value)));
  }, [min, max]);

  return {
    count,
    increment,
    decrement,
    reset,
    set,
    canIncrement: count < max,
    canDecrement: count > min,
  };
}
