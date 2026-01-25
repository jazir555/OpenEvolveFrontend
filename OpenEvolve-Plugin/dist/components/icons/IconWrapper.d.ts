import { default as React } from 'react';
/**
 * IconWrapper - Wraps icon components to accept className and other props
 *
 * This component wraps icon components (which may not accept className directly)
 * and forwards all props to them, enabling consistent styling.
 */
export declare function IconWrapper({ icon: Icon, className, ...props }: {
    icon: React.ComponentType<any>;
    className?: string;
    [key: string]: any;
}): import("react/jsx-runtime").JSX.Element;
