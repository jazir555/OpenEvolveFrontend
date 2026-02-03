/**
 * Kbd Component
 * Keyboard shortcut display
 */

interface KbdProps {
  children: string;
  className?: string;
}

export function Kbd({ children, className = '' }: KbdProps) {
  return (
    <kbd
      className={`inline-flex items-center px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-300 rounded-lg dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600 ${className}`}
    >
      {children}
    </kbd>
  );
}
