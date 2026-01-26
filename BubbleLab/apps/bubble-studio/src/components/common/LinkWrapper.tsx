/**
 * LinkWrapper Component
 * Consistent link styling with external/internal handling
 */

import { Link } from '@tanstack/react-router';

interface LinkWrapperProps {
  to: string;
  children: React.ReactNode;
  external?: boolean;
  variant?: 'primary' | 'secondary' | 'muted';
  underline?: boolean;
  className?: string;
  onClick?: () => void;
}

const variantStyles = {
  primary: 'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300',
  secondary: 'text-gray-900 hover:text-gray-700 dark:text-white dark:hover:text-gray-300',
  muted: 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300',
};

export function LinkWrapper({
  to,
  children,
  external = false,
  variant = 'primary',
  underline = true,
  className = '',
  onClick,
}: LinkWrapperProps) {
  const baseClasses = `${variantStyles[variant]} ${underline ? 'underline' : ''} transition-colors duration-150 ${className}`;

  if (external) {
    return (
      <a
        href={to}
        target="_blank"
        rel="noopener noreferrer"
        className={baseClasses}
        onClick={onClick}
      >
        {children}
      </a>
    );
  }

  return (
    <Link to={to} className={baseClasses} onClick={onClick}>
      {children}
    </Link>
  );
}
