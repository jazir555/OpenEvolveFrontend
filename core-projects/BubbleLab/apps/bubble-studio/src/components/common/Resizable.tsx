/**
 * Resizable Component
 * Resizable panel with drag handle
 */

import { useState, useRef, useCallback, useEffect } from 'react';

interface ResizableProps {
  children: React.ReactNode;
  initialWidth?: number;
  minWidth?: number;
  maxWidth?: number;
  direction?: 'horizontal' | 'vertical';
  className?: string;
}

export function Resizable({
  children,
  initialWidth = 300,
  minWidth = 100,
  maxWidth = 800,
  direction = 'horizontal',
  className = '',
}: ResizableProps) {
  const [width, setWidth] = useState(initialWidth);
  const [isResizing, setIsResizing] = useState(false);
  const resizeHandleRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsResizing(true);
    e.preventDefault();
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return;

      const newWidth = direction === 'horizontal' ? e.clientX : e.clientY;
      const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
      setWidth(clampedWidth);
    },
    [isResizing, minWidth, maxWidth, direction]
  );

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing, handleMouseMove, handleMouseUp]);

  return (
    <div
      className={`relative ${className}`}
      style={{
        width: direction === 'horizontal' ? width : '100%',
        height: direction === 'vertical' ? width : '100%',
      }}
    >
      {children}

      {/* Resize Handle */}
      <div
        ref={resizeHandleRef}
        onMouseDown={handleMouseDown}
        className={`
          absolute top-0 bottom-0 cursor-col-resize hover:bg-blue-500
          ${direction === 'horizontal' ? 'right-0 w-1' : 'bottom-0 h-1 w-full cursor-row-resize'}
          ${isResizing ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'}
          transition-colors
        `}
      />
    </div>
  );
}
