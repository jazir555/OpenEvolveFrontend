/**
 * Tag/Chip Component
 * Categorization tag
 */

interface TagProps {
  children: React.ReactNode;
  variant?: 'gray' | 'blue' | 'green' | 'red' | 'yellow' | 'purple';
  size?: 'sm' | 'md' | 'lg';
  onRemove?: () => void;
}

export function Tag({ children, variant = 'gray', size = 'md', onRemove }: TagProps) {
  const variantStyles = {
    gray: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
    blue: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    green: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    red: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    yellow: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    purple: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
  };

  const sizeStyles = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-0.5',
    lg: 'text-base px-3 py-1',
  };

  return (
    <span className={`inline-flex items-center rounded-full ${variantStyles[variant]} ${sizeStyles[size]}`}>
      {children}
      {onRemove && (
        <button
          onClick={onRemove}
          className="ml-1 text-gray-500 hover:text-gray-700 dark:text-gray-400"
        >
          <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l-4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      )}
    </span>
  );
}
