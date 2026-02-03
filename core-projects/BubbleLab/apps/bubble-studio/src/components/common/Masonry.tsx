/**
 * Masonry Component
 * Masonry grid layout for varying height items
 */

import { useRef, useEffect, useState, ReactNode } from 'react';

interface MasonryProps {
  items: ReactNode[];
  columns?: number;
  gap?: number;
  className?: string;
}

export function Masonry({
  items,
  columns = 3,
  gap = 16,
  className = '',
}: MasonryProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [columnHeights, setColumnHeights] = useState<number[]>([]);

  useEffect(() => {
    const calculateHeights = () => {
      if (!containerRef.current) return;

      const columnElements = containerRef.current.children;
      const heights: number[] = [];

      for (let i = 0; i < columns; i++) {
        if (columnElements[i]) {
          heights.push(columnElements[i].getBoundingClientRect().height);
        }
      }

      setColumnHeights(heights);
    };

    calculateHeights();

    const resizeObserver = new ResizeObserver(calculateHeights);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, [items, columns]);

  // Distribute items to columns
  const columnItems: ReactNode[][] = Array.from({ length: columns }, () => []);
  items.forEach((item, index) => {
    const columnIndex = index % columns;
    columnItems[columnIndex].push(item);
  });

  return (
    <div
      ref={containerRef}
      className={`grid gap-${gap / 4} ${className}`}
      style={{
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
      }}
    >
      {columnItems.map((column, columnIndex) => (
        <div key={columnIndex} className="flex flex-col gap-4">
          {column.map((item, itemIndex) => (
            <div key={itemIndex}>{item}</div>
          ))}
        </div>
      ))}
    </div>
  );
}
