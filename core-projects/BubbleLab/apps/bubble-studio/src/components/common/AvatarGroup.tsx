/**
 * AvatarGroup Component
 * Group of overlapping avatars
 */

import { Avatar } from './Avatar';

interface AvatarGroupProps {
  avatars: Array<{
    src?: string;
    name?: string;
    alt?: string;
  }>;
  max?: number;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

export function AvatarGroup({
  avatars,
  max = 3,
  size = 'md',
  className = '',
}: AvatarGroupProps) {
  const visibleAvatars = avatars.slice(0, max);
  const remainingCount = Math.max(0, avatars.length - max);

  const sizeStyles = {
    sm: '-space-x-2',
    md: '-space-x-3',
    lg: '-space-x-4',
    xl: '-space-x-5',
  };

  return (
    <div className={`flex ${sizeStyles[size]} ${className}`}>
      {visibleAvatars.map((avatar, index) => (
        <div
          key={index}
          className="inline-block border-2 border-white dark:border-gray-800 rounded-full"
          style={{ zIndex: avatars.length - index }}
        >
          <Avatar src={avatar.src} name={avatar.name} alt={avatar.alt} size={size} />
        </div>
      ))}

      {remainingCount > 0 && (
        <div
          className="inline-flex items-center justify-center w-8 h-8 rounded-full border-2 border-white dark:border-gray-800 bg-gray-200 dark:bg-gray-700 text-xs font-medium text-gray-600 dark:text-gray-300"
          style={{ zIndex: 0 }}
        >
          +{remainingCount}
        </div>
      )}
    </div>
  );
}
