/**
 * Safe Context Utilities
 * 
 * Provides React context with comprehensive error handling
 */

import React, { createContext, useContext, Context, ReactNode } from 'react';
import { errorLogger } from './errorLogging';

/**
 * Safe context provider with error handling
 */
export function createSafeContext<T>(
  defaultValue: T,
  contextName: string = 'SafeContext'
): [() => T, React.FC<{ children: ReactNode; value?: T }>] {
  const Context = createContext<T>(defaultValue);

  const SafeContextProvider: React.FC<{ children: ReactNode; value?: T }> = ({ 
    children, 
    value = defaultValue 
  }) => {
    return (
      <Context.Provider value={value}>
        <SafeContextErrorBoundary contextName={contextName}>
          {children}
        </SafeContextErrorBoundary>
      </Context.Provider>
    );
  };

  const useSafeContext = (): T => {
    const context = useContext(Context);
    
    if (context === undefined) {
      const error = new Error(`use${contextName} must be used within a ${contextName}Provider`);
      errorLogger.logError(error, 'error', {
        component: contextName,
        function: 'useSafeContext',
        additionalData: { contextName }
      });
      throw error;
    }
    
    return context;
  };

  return [useSafeContext, SafeContextProvider];
}

/**
 * Error boundary specifically for contexts
 */
const SafeContextErrorBoundary: React.FC<{ 
  children: ReactNode; 
  contextName: string 
}> = ({ children, contextName }) => {
  // Using the ComponentErrorBoundary from shared components
  const ComponentErrorBoundary = React.lazy(() => 
    import('@/components/shared/ComponentErrorBoundary').then(module => 
      ({ default: module.ComponentErrorBoundary })
    )
  );

  return (
    <React.Suspense fallback={<div>Loading context...</div>}>
      <ComponentErrorBoundary label={`${contextName} Context`}>
        {children}
      </ComponentErrorBoundary>
    </React.Suspense>
  );
};

/**
 * Safe consumer with error handling
 */
export function createSafeConsumer<T>(
  Context: Context<T>,
  contextName: string
): (selector?: (value: T) => any) => any {
  return (selector?: (value: T) => any) => {
    const context = useContext(Context);
    
    if (context === undefined) {
      const error = new Error(`use${contextName} must be used within a ${contextName}Provider`);
      errorLogger.logError(error, 'error', {
        component: contextName,
        function: 'createSafeConsumer',
        additionalData: { contextName, hasSelector: !!selector }
      });
      throw error;
    }
    
    try {
      return selector ? selector(context) : context;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: contextName, 
          function: 'createSafeConsumer.selector', 
          additionalData: { contextName, hasSelector: !!selector } 
        }
      );
      return context;
    }
  };
}

/**
 * Safe context hook with error handling
 */
export function useSafeContextHook<T>(
  Context: Context<T>,
  contextName: string,
  options: {
    defaultValue?: T;
    onError?: (error: Error) => void;
  } = {}
): T {
  const { defaultValue, onError } = options;
  
  try {
    const context = useContext(Context);
    
    if (context === undefined) {
      const error = new Error(`use${contextName} must be used within a ${contextName}Provider`);
      errorLogger.logError(error, 'error', {
        component: contextName,
        function: 'useSafeContextHook',
        additionalData: { contextName }
      });
      
      onError?.(error);
      
      if (defaultValue !== undefined) {
        return defaultValue;
      }
      
      throw error;
    }
    
    return context;
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: contextName, 
        function: 'useSafeContextHook', 
        additionalData: { contextName } 
      }
    );
    
    onError?.(typedError);
    
    if (defaultValue !== undefined) {
      return defaultValue;
    }
    
    throw typedError;
  }
}

/**
 * Safe context provider component with error handling
 */
export function SafeContextProvider<T>({
  Context,
  value,
  children,
  contextName = 'AnonymousContext'
}: {
  Context: Context<T>;
  value: T;
  children: ReactNode;
  contextName?: string;
}): JSX.Element {
  try {
    return (
      <Context.Provider value={value}>
        {children}
      </Context.Provider>
    );
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: contextName, 
        function: 'SafeContextProvider', 
        additionalData: { contextName } 
      }
    );
    
    // Fallback UI in case of provider error
    return (
      <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
        <div className="font-semibold">{contextName} Provider Error</div>
        <div className="mt-1">Context provider failed to initialize.</div>
      </div>
    );
  }
}

/**
 * Safe context consumer component with error handling
 */
export function SafeContextConsumer<T>({
  Context,
  children,
  contextName = 'AnonymousContext',
  fallback
}: {
  Context: Context<T>;
  children: (value: T) => ReactNode;
  contextName?: string;
  fallback?: ReactNode;
}): JSX.Element {
  try {
    const context = useContext(Context);
    
    if (context === undefined) {
      const error = new Error(`SafeContextConsumer must be used within a ${contextName}Provider`);
      errorLogger.logError(error, 'error', {
        component: contextName,
        function: 'SafeContextConsumer',
        additionalData: { contextName }
      });
      
      return fallback || (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
          <div className="font-semibold">{contextName} Consumer Error</div>
          <div className="mt-1">Context consumer not wrapped in provider.</div>
        </div>
      );
    }
    
    return children(context);
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: contextName, 
        function: 'SafeContextConsumer', 
        additionalData: { contextName } 
      }
    );
    
    return fallback || (
      <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
        <div className="font-semibold">{contextName} Consumer Error</div>
        <div className="mt-1">Error consuming context value.</div>
      </div>
    );
  }
}