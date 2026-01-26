/**
 * Icon Component
 * Icon wrapper for consistency
 */

interface IconProps {
  name: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function Icon({ name, size = 'md', className = '' }: IconProps) {
  const sizeStyles = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6',
  };

  // This would load icons from Heroicons
  // For now, returning a placeholder
  return (
    <svg className={sizeStyles[size]} fill="currentColor" viewBox="0 0 20 20">
      <path d="M10 2a6 6 0 11-6 6 6 0 016 6z" />
    </svg>
  );
}
