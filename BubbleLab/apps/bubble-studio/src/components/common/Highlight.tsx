/**
 * Highlight Component
 * Text highlighting with search terms
 */

interface HighlightProps {
  text: string;
  highlight: string;
  className?: string;
}

export function Highlight({ text, highlight, className = '' }: HighlightProps) {
  if (!highlight.trim()) {
    return <span className={className}>{text}</span>;
  }

  const regex = new RegExp(`(${highlight})`, 'gi');
  const parts = text.split(regex);

  return (
    <span className={className}>
      {parts.map((part, index) =>
        regex.test(part) ? (
          <mark key={index} className="bg-yellow-200 dark:bg-yellow-900/30 text-gray-900 dark:text-white rounded px-0.5">
            {part}
          </mark>
        ) : (
          part
        )
      )}
    </span>
  );
}
