/**
 * useIntersectionObserver Hook
 * Detect when element enters/exits viewport
 */

import { useState, useEffect, useRef } from 'react';

interface UseIntersectionObserverOptions {
  threshold?: number | number[];
  rootMargin?: string;
  root?: Element | null;
  triggerOnce?: boolean;
}

export function useIntersectionObserver(
  options: UseIntersectionObserverOptions = {}
): [React.RefObject<HTMLDivElement>, boolean] {
  const { threshold = 0, rootMargin = '0px', root = null, triggerOnce = false } = options;

  const [isIntersecting, setIsIntersecting] = useState(false);
  const [hasIntersected, setHasIntersected] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    // If triggerOnce and already intersected, don't observe
    if (triggerOnce && hasIntersected) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        const isElementIntersecting = entry.isIntersecting;

        setIsIntersecting(isElementIntersecting);

        if (isElementIntersecting && triggerOnce) {
          setHasIntersected(true);
        }
      },
      { threshold, rootMargin, root }
    );

    observer.observe(node);

    return () => {
      observer.disconnect();
    };
  }, [threshold, rootMargin, root, triggerOnce, hasIntersected]);

  return [ref, isIntersecting];
}

/**
 * useInViewport Hook
 * Simplified hook to check if element is in viewport
 */
export function useInViewport(): [React.RefObject<HTMLDivElement>, boolean] {
  return useIntersectionObserver({ threshold: 0 });
}

/**
 * useOnScreen Hook
 * Trigger callback when element enters viewport
 */
export function useOnScreen(
  callback: () => void,
  options: UseIntersectionObserverOptions = {}
): React.RefObject<HTMLDivElement> {
  const [ref, isIntersecting] = useIntersectionObserver(options);

  useEffect(() => {
    if (isIntersecting) {
      callback();
    }
  }, [isIntersecting, callback]);

  return ref;
}
