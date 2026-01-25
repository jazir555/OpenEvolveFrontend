/**
 * BubbleLab Compatible Card Component
 * 
 * A React card component that follows BubbleLab's design system
 */

import React from 'react';

interface BubbleCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

const BubbleCard: React.FC<BubbleCardProps> = ({
  children,
  className = '',
  ...props
}) => {
  const classes = `rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm ${className}`;

  return (
    <div className={classes} {...props}>
      {children}
    </div>
  );
};

export default BubbleCard;