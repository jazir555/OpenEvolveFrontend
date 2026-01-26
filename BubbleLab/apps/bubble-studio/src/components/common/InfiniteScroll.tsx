/**
 * InfiniteScroll Component
 * Load more content as user scrolls
 */

import { useEffect, useRef, ReactNode } from 'react';
import { useIntersectionObserver } from '../../hooks/useIntersectionObserver';

interface InfiniteScrollProps {
  children: ReactNode;
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
  threshold?: number;
  className?: string;
}

export function InfiniteScroll({
  children,
  hasMore,
  isLoading,
  onLoadMore,
  threshold = 0,
  className = '',
}: InfiniteScrollProps) {
  const [loadMoreRef, isIntersecting] = useIntersectionObserver({ threshold });

  useEffect(() => {
    if (isIntersecting && hasMore && !isLoading) {
      onLoadMore();
    }
  }, [isIntersecting, hasMore, isLoading, onLoadMore]);

  return (
    <div className={className}>
      {children}

      {/* Loading trigger */}
      <div ref={loadMoreRef} className="py-4">
        {isLoading && (
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          </div>
        )}

        {!hasMore && (
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            No more items to load
          </p>
        )}
      </div>
    </div>
  );
}
