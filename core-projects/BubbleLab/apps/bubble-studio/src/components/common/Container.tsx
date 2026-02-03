/**
 * Container Component
 * Constrained width container
 */

interface ContainerProps {
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  centered?: boolean;
  className?: string;
}

const sizeStyles = {
  sm: 'max-w-screen-sm',
  md: 'max-w-screen-md',
  lg: 'max-w-screen-lg',
  xl: 'max-w-screen-xl',
  full: 'max-w-full',
};

export function Container({
  children,
  size = 'lg',
  centered = false,
  className = '',
}: ContainerProps) {
  return (
    <div
      className={`${sizeStyles[size]} ${
        centered ? 'mx-auto' : ''
      } px-4 sm:px-6 lg:px-8 ${className}`}
    >
      {children}
    </div>
  );
}
