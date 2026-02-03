/**
 * Rating Component
 * Star rating input and display
 */

interface RatingProps {
  value: number;
  onChange?: (value: number) => void;
  max?: number;
  readonly?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function Rating({
  value,
  onChange,
  max = 5,
  readonly = false,
  size = 'md',
  className = '',
}: RatingProps) {
  const sizeStyles = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  };

  const handleStarClick = (index: number) => {
    if (readonly || !onChange) return;
    onChange(index + 1);
  };

  return (
    <div className={`flex items-center gap-1 ${className}`}>
      {Array.from({ length: max }).map((_, index) => {
        const filled = index < value;
        return (
          <button
            key={index}
            type="button"
            onClick={() => handleStarClick(index)}
            disabled={readonly}
            className={`${readonly ? 'cursor-default' : 'cursor-pointer'} ${
              readonly ? '' : 'hover:scale-110'
            } transition-transform`}
          >
            <svg
              className={`${sizeStyles[size]} ${
                filled
                  ? 'text-yellow-400 fill-current'
                  : 'text-gray-300 dark:text-gray-600'
              }`}
              viewBox="0 0 20 20"
            >
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
          </button>
        );
      })}
      {!readonly && onChange && (
        <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
          {value}/{max}
        </span>
      )}
    </div>
  );
}
