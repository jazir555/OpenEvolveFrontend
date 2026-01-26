/**
 * Divider Component
 * Visual separator
 */

interface DividerProps {
  orientation?: 'horizontal' | 'vertical';
  variant?: 'solid' | 'dashed';
  thickness?: 'thin' | 'medium' | 'thick';
  className?: string;
}

const thicknessStyles = {
  thin: 'border-t',
  medium: 'border-t-2',
  thick: 'border-t-4',
};

const verticalThicknessStyles = {
  thin: 'border-l',
  medium: 'border-l-2',
  thick: 'border-l-4',
};

export function Divider({
  orientation = 'horizontal',
  variant = 'solid',
  thickness = 'thin',
  className = '',
}: DividerProps) {
  if (orientation === 'vertical') {
    return (
      <div
        className={`h-full ${verticalThicknessStyles[thickness]} border-gray-300 dark:border-gray-700 ${
          variant === 'dashed' ? 'border-dashed' : ''
        } ${className}`}
      />
    );
  }

  return (
    <div
      className={`w-full ${thicknessStyles[thickness]} border-gray-300 dark:border-gray-700 my-4 ${
        variant === 'dashed' ? 'border-dashed' : ''
      } ${className}`}
    />
  );
}
