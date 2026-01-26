/**
 * Avatar Component
 * User avatar with fallback
 */

interface AvatarProps {
  src?: string;
  alt?: string;
  name?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

export function Avatar({ src, alt, name, size = 'md', className = '' }: AvatarProps) {
  const sizeStyles = {
    sm: 'h-6 w-6 text-xs',
    md: 'h-8 w-8 text-sm',
    lg: 'h-10 w-10 text-base',
    xl: 'h-12 w-12 text-lg',
  };

  const initials = name
    ? name
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2)
    : '??';

  return (
    <div className={`relative inline-flex items-center justify-center rounded-full bg-gray-300 ${sizeStyles[size]} ${className}`}>
      {src ? (
        <img
          src={src}
          alt={alt || name}
          className="h-full w-full rounded-full object-cover"
        />
      ) : (
        <span className="font-medium text-gray-700">{initials}</span>
      )}
    </div>
  );
}
