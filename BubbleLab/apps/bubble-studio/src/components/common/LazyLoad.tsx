/**
 * LazyLoad Component
 * Lazy load content when it enters viewport
 */

import { ReactNode, useState, useRef } from 'react';
import { useIntersectionObserver } from '../../hooks/useIntersectionObserver';

interface LazyLoadProps {
  children: ReactNode;
  fallback?: ReactNode;
  offset?: number;
  className?: string;
}

export function LazyLoad({
  children,
  fallback = null,
  offset = 100,
  className = '',
}: LazyLoadProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [ref, isInView] = useIntersectionObserver({
    rootMargin: `${offset}px`,
  });

  // Once loaded, stay loaded
  if (isLoaded) {
    return <>{children}</>;
  }

  // Load when in view
  if (isInView && !isLoaded) {
    setIsLoaded(true);
    return <>{children}</>;
  }

  return (
    <div ref={ref} className={className}>
      {fallback}
    </div>
  );
}

/**
 * LazyLoadImage Component
 * Lazy load images
 */
interface LazyLoadImageProps {
  src: string;
  alt: string;
  className?: string;
  placeholder?: string;
}

export function LazyLoadImage({
  src,
  alt,
  className = '',
  placeholder = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"%3E%3Crect width="400" height="300" fill="%23e5e7eb"/%3E%3C/svg%3E',
}: LazyLoadImageProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [ref, isInView] = useIntersectionObserver();

  return (
    <img
      ref={ref}
      src={isLoaded || isInView ? src : placeholder}
      alt={alt}
      className={className}
      onLoad={() => setIsLoaded(true)}
    />
  );
}
