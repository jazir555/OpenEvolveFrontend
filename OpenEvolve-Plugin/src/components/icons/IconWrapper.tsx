import React from 'react';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

/**
 * IconWrapper - Wraps icon components to accept className and other props
 *
 * This component wraps icon components (which may not accept className directly)
 * and forwards all props to them, enabling consistent styling.
 */
function IconWrapperBase({
  icon: Icon,
  className = '',
  ...props
}: {
  icon: React.ComponentType<any>;
  className?: string;
  [key: string]: any;
}) {
  return <Icon className={className} {...props} />;
}

export const IconWrapper = withComponentBoundary(IconWrapperBase, 'IconWrapper');
