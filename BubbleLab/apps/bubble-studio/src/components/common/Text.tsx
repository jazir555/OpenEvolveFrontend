/**
 * Text Component
 * Consistent text styling
 */

interface TextProps {
  children: React.ReactNode;
  variant?: 'h1' | 'h2' | 'h3' | 'h4' | 'body' | 'small' | 'caption';
  weight?: 'normal' | 'medium' | 'semibold' | 'bold';
  color?: 'primary' | 'secondary' | 'muted' | 'danger' | 'success';
  className?: string;
}

const variantStyles = {
  h1: 'text-3xl font-bold tracking-tight',
  h2: 'text-2xl font-semibold tracking-tight',
  h3: 'text-xl font-semibold tracking-tight',
  h4: 'text-lg font-medium',
  body: 'text-base',
  small: 'text-sm',
  caption: 'text-xs',
};

const colorStyles = {
  primary: 'text-gray-900 dark:text-white',
  secondary: 'text-gray-700 dark:text-gray-300',
  muted: 'text-gray-500 dark:text-gray-400',
  danger: 'text-red-600 dark:text-red-400',
  success: 'text-green-600 dark:text-green-400',
};

const weightStyles = {
  normal: 'font-normal',
  medium: 'font-medium',
  semibold: 'font-semibold',
  bold: 'font-bold',
};

export function Text({
  children,
  variant = 'body',
  weight = 'normal',
  color = 'primary',
  className = '',
}: TextProps) {
  const Tag = variant === 'h1' || variant === 'h2' || variant === 'h3' || variant === 'h4'
    ? (variant as 'h1' | 'h2' | 'h3' | 'h4')
    : 'p';

  return (
    <Tag className={`${variantStyles[variant]} ${colorStyles[color]} ${weightStyles[weight]} ${className}`}>
      {children}
    </Tag>
  );
}
