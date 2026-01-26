/**
 * useArray Hook
 * Manage array state
 */

import { useCallback, useState } from 'react';

export function useArray<T>(initialValue: T[] = []) {
  const [array, setArray] = useState<T[]>(initialValue);

  const push = useCallback((item: T) => {
    setArray((prev) => [...prev, item]);
  }, []);

  const unshift = useCallback((item: T) => {
    setArray((prev) => [item, ...prev]);
  }, []);

  const pop = useCallback(() => {
    setArray((prev) => prev.slice(0, -1));
  }, []);

  const shift = useCallback(() => {
    setArray((prev) => prev.slice(1));
  }, []);

  const splice = useCallback((start: number, deleteCount = 0, ...items: T[]) => {
    setArray((prev) => {
      const copy = [...prev];
      copy.splice(start, deleteCount, ...items);
      return copy;
    });
  }, []);

  const remove = useCallback((index: number) => {
    setArray((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const move = useCallback((from: number, to: number) => {
    setArray((prev) => {
      const copy = [...prev];
      const [item] = copy.splice(from, 1);
      copy.splice(to, 0, item);
      return copy;
    });
  }, []);

  const clear = useCallback(() => {
    setArray([]);
  }, []);

  const update = useCallback((index: number, item: T) => {
    setArray((prev) => {
      const copy = [...prev];
      copy[index] = item;
      return copy;
    });
  }, []);

  return {
    array,
    setArray,
    push,
    unshift,
    pop,
    shift,
    splice,
    remove,
    move,
    clear,
    update,
  };
}
