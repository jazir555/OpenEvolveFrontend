/**
 * Paper Component
 * Elevated surface container
 */

interface PaperProps {
  children: React.ReactNode;
  elevation?: 'none' | 'sm' | 'md' | 'lg';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
  className?: string;
}

const elevationStyles = {
  none: '',
  sm: 'shadow-sm',
  md: 'shadow-md',
  lg: 'shadow-lg',
};

const paddingStyles = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
};

const roundedStyles = {
  none: '',
  sm: 'rounded-sm',
  md: 'rounded-md',
  lg: 'rounded-lg',
  full: 'rounded-full',
};

export function Paper({
  children,
  elevation = 'md',
  padding = 'md',
  rounded = 'md',
  className = '',
}: PaperProps) {
  return (
    <div
      className={`bg-white dark:bg-gray-800 ${elevationStyles[elevation]} ${paddingStyles[padding]} ${roundedStyles[rounded]} ${className}`}
    >
      {children}
    </div>
  );
}
