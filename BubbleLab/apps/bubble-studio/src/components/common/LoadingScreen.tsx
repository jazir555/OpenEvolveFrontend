/**
 * LoadingScreen Component
 * Full-screen loading indicator
 */

import { Spinner } from './Spinner';

interface LoadingScreenProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function LoadingScreen({ message = 'Loading...', size = 'lg' }: LoadingScreenProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-white dark:bg-gray-900">
      <div className="text-center">
        <Spinner size={size} />
        {message && (
          <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            {message}
          </p>
        )}
      </div>
    </div>
  );
}
