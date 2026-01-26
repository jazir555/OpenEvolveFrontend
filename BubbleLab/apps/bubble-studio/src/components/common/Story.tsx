/**
 * Story Component
 * Social media style story bubbles
 */

interface Story {
  id: string;
  name: string;
  avatar?: string;
  seen?: boolean;
  onClick: () => void;
}

interface StoryProps {
  stories: Story[];
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function Story({ stories, size = 'md', className = '' }: StoryProps) {
  const sizeStyles = {
    sm: 'w-10 h-10',
    md: 'w-14 h-14',
    lg: 'w-20 h-20',
  };

  return (
    <div className={`flex gap-4 overflow-x-auto pb-2 ${className}`}>
      {stories.map((story) => (
        <button
          key={story.id}
          onClick={story.onClick}
          className="flex-shrink-0 group"
        >
          <div className={`${sizeStyles[size]} rounded-full p-1 ${
            story.seen
              ? 'bg-gray-300 dark:bg-gray-700'
              : 'bg-gradient-to-tr from-yellow-400 via-pink-500 to-purple-500'
          }`}>
            <div className={`${sizeStyles[size]} rounded-full border-2 border-white dark:border-gray-800 overflow-hidden`}>
              {story.avatar ? (
                <img
                  src={story.avatar}
                  alt={story.name}
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-200"
                />
              ) : (
                <div className="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-500 dark:text-gray-400 text-xs">
                  {story.name.charAt(0)}
                </div>
              )}
            </div>
          </div>
          <p className="text-xs text-center mt-1 text-gray-600 dark:text-gray-400 max-w-[60px] truncate">
            {story.name}
          </p>
        </button>
      ))}
    </div>
  );
}
