/**
 * Safe Routing Utilities
 * 
 * Provides routing utilities with comprehensive error handling
 */

import { NavigateFunction, Location, To } from 'react-router-dom';
import { errorLogger } from './errorLogging';

/**
 * Safe navigation wrapper with error handling
 */
export function createSafeNavigate(navigate: NavigateFunction): NavigateFunction {
  return ((to: To, options?: { replace?: boolean; state?: any }) => {
    try {
      navigate(to, options);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeRouting', 
          function: 'navigate', 
          additionalData: { to, options } 
        }
      );
    }
  }) as NavigateFunction;
}

/**
 * Safe route change handler with error handling
 */
export function handleRouteChange(
  navigate: NavigateFunction,
  to: To,
  options?: { replace?: boolean; state?: any },
  onError?: (error: Error) => void
) {
  try {
    navigate(to, options);
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeRouting', 
        function: 'handleRouteChange', 
        additionalData: { to, options } 
      }
    );
    
    onError?.(typedError);
  }
}

/**
 * Safe route guard with error handling
 */
export function withRouteGuard(
  condition: boolean,
  fallbackRoute: To,
  navigate: NavigateFunction,
  options?: { replace?: boolean; state?: any }
) {
  if (!condition) {
    try {
      navigate(fallbackRoute, options);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeRouting', 
          function: 'withRouteGuard', 
          additionalData: { condition, fallbackRoute, options } 
        }
      );
    }
  }
}

/**
 * Safe route loader with error handling
 */
export async function safeRouteLoader(
  loaderFn: () => Promise<any>,
  options?: {
    onError?: (error: Error) => void;
    fallbackValue?: any;
  }
): Promise<any> {
  const { onError, fallbackValue } = options || {};
  
  try {
    const result = await loaderFn();
    return result;
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeRouting', 
        function: 'safeRouteLoader', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    
    onError?.(typedError);
    
    return fallbackValue;
  }
}

/**
 * Safe route action with error handling
 */
export async function safeRouteAction(
  actionFn: () => Promise<any>,
  options?: {
    onError?: (error: Error) => void;
    onSuccess?: (result: any) => void;
    fallbackValue?: any;
  }
): Promise<any> {
  const { onError, onSuccess, fallbackValue } = options || {};
  
  try {
    const result = await actionFn();
    onSuccess?.(result);
    return result;
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeRouting', 
        function: 'safeRouteAction', 
        additionalData: { hasOnError: !!onError, hasOnSuccess: !!onSuccess } 
      }
    );
    
    onError?.(typedError);
    
    return fallbackValue;
  }
}

/**
 * Safe route matcher with error handling
 */
export function safeMatchRoutes(
  routes: Array<{ path: string; element: any }>,
  pathname: string,
  options?: {
    onError?: (error: Error) => void;
  }
): typeof routes[0] | undefined {
  const { onError } = options || {};
  
  try {
    return routes.find(route => {
      try {
        // Simple path matching (could be enhanced with more sophisticated matching)
        return pathname.startsWith(route.path) || pathname === route.path;
      } catch (matchError) {
        errorLogger.logError(
          matchError instanceof Error ? matchError : new Error(String(matchError)),
          'error',
          { 
            component: 'SafeRouting', 
            function: 'safeMatchRoutes.matcher', 
            additionalData: { pathname, routePath: route.path } 
          }
        );
        return false;
      }
    });
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeRouting', 
        function: 'safeMatchRoutes', 
        additionalData: { pathname, routeCount: routes.length } 
      }
    );
    
    onError?.(typedError);
    
    return undefined;
  }
}

/**
 * Safe route parameters parser with error handling
 */
export function safeParseRouteParams(
  params: URLSearchParams | string,
  options?: {
    onError?: (error: Error) => void;
    fallbackValue?: Record<string, string>;
  }
): Record<string, string> {
  const { onError, fallbackValue = {} } = options || {};
  
  try {
    const searchParams = typeof params === 'string' ? new URLSearchParams(params) : params;
    const result: Record<string, string> = {};
    
    for (const [key, value] of searchParams) {
      result[key] = value;
    }
    
    return result;
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeRouting', 
        function: 'safeParseRouteParams', 
        additionalData: { paramsType: typeof params } 
      }
    );
    
    onError?.(typedError);
    
    return fallbackValue;
  }
}

/**
 * Safe route state manager with error handling
 */
export class SafeRouteStateManager {
  private state: Record<string, any> = {};
  private listeners: Array<(state: Record<string, any>) => void> = [];

  constructor(initialState: Record<string, any> = {}) {
    try {
      this.state = { ...initialState };
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeRouting', 
          function: 'SafeRouteStateManager.constructor', 
          additionalData: { initialState } 
        }
      );
      this.state = {};
    }
  }

  setState(newState: Record<string, any>, options?: { silent?: boolean }) {
    try {
      this.state = { ...this.state, ...newState };
      
      if (!options?.silent) {
        this.notifyListeners();
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeRouting', 
          function: 'SafeRouteStateManager.setState', 
          additionalData: { newState, silent: options?.silent } 
        }
      );
    }
  }

  getState(): Record<string, any> {
    try {
      return { ...this.state };
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeRouting', 
          function: 'SafeRouteStateManager.getState' 
        }
      );
      return {};
    }
  }

  subscribe(listener: (state: Record<string, any>) => void): () => void {
    try {
      this.listeners.push(listener);
      
      return () => {
        try {
          this.listeners = this.listeners.filter(l => l !== listener);
        } catch (unsubscribeError) {
          errorLogger.logError(
            unsubscribeError instanceof Error ? unsubscribeError : new Error(String(unsubscribeError)),
            'error',
            { 
              component: 'SafeRouting', 
              function: 'SafeRouteStateManager.unsubscribe' 
            }
          );
        }
      };
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeRouting', 
          function: 'SafeRouteStateManager.subscribe' 
        }
      );
      
      return () => {}; // Return noop unsubscribe function
    }
  }

  private notifyListeners() {
    try {
      this.listeners.forEach(listener => {
        try {
          listener(this.getState());
        } catch (listenerError) {
          errorLogger.logError(
            listenerError instanceof Error ? listenerError : new Error(String(listenerError)),
            'error',
            { 
              component: 'SafeRouting', 
              function: 'SafeRouteStateManager.notifyListeners.listener' 
            }
          );
        }
      });
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeRouting', 
          function: 'SafeRouteStateManager.notifyListeners' 
        }
      );
    }
  }
}

/**
 * Safe location watcher with error handling
 */
export function safeWatchLocation(
  callback: (location: Location) => void,
  options?: {
    onError?: (error: Error) => void;
  }
): () => void {
  const { onError } = options || {};

  try {
    // Check if we're in a browser environment
    if (typeof window === 'undefined') {
      throw new Error('safeWatchLocation can only be used in a browser environment');
    }

    const handleLocationChange = () => {
      try {
        // Get current location from window.location
        const currentLocation: Location = {
          pathname: window.location.pathname,
          search: window.location.search,
          hash: window.location.hash,
          state: null,
          key: 'default'
        } as Location;

        callback(currentLocation);
      } catch (callbackError) {
        const typedError = callbackError instanceof Error ? callbackError : new Error(String(callbackError));

        errorLogger.logError(
          typedError,
          'error',
          {
            component: 'SafeRouting',
            function: 'safeWatchLocation.callback',
            additionalData: { hasOnError: !!onError }
          }
        );

        onError?.(typedError);
      }
    };

    // Listen to popstate events (back/forward navigation)
    window.addEventListener('popstate', handleLocationChange);

    // Also listen to hashchange and pushstate/hashchange custom events
    window.addEventListener('hashchange', handleLocationChange);

    // Listen for custom events from programmatic navigation
    const customEventHandler = (event: Event) => {
      handleLocationChange();
    };
    window.addEventListener('pushstate', customEventHandler);
    window.addEventListener('replacestate', customEventHandler);

    // Initial call with current location
    handleLocationChange();

    // Return cleanup function
    return () => {
      try {
        window.removeEventListener('popstate', handleLocationChange);
        window.removeEventListener('hashchange', handleLocationChange);
        window.removeEventListener('pushstate', customEventHandler);
        window.removeEventListener('replacestate', customEventHandler);
      } catch (cleanupError) {
        errorLogger.logError(
          cleanupError instanceof Error ? cleanupError : new Error(String(cleanupError)),
          'error',
          {
            component: 'SafeRouting',
            function: 'safeWatchLocation.cleanup'
          }
        );
      }
    };
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));

    errorLogger.logError(
      typedError,
      'error',
      {
        component: 'SafeRouting',
        function: 'safeWatchLocation',
        additionalData: { hasOnError: !!onError }
      }
    );

    onError?.(typedError);

    return () => {}; // Return noop cleanup function
  }
}

/**
 * Navigate to a path with proper state preservation
 */
export interface NavigateOptions {
  replace?: boolean;
  state?: any;
  preserveQuery?: boolean;
}

export function navigate(path: string, options?: NavigateOptions): void {
  try {
    if (typeof window === 'undefined') {
      throw new Error('navigate can only be used in a browser environment');
    }

    const { replace = false, state, preserveQuery = false } = options || {};

    // Build the full URL
    let url = path;

    // Handle relative vs absolute paths
    if (!path.startsWith('http') && !path.startsWith('/')) {
      // Relative path - resolve against current location
      const currentPath = window.location.pathname;
      const basePath = currentPath.substring(0, currentPath.lastIndexOf('/'));
      url = `${basePath}/${path}`;
    }

    // Preserve query parameters if requested
    if (preserveQuery && window.location.search) {
      const separator = url.includes('?') ? '&' : '?';
      url = `${url}${separator}${window.location.search.substring(1)}`;
    }

    // Use History API for navigation
    if (replace) {
      window.history.replaceState(state || null, '', url);
    } else {
      window.history.pushState(state || null, '', url);
    }

    // Dispatch custom event for safeWatchLocation to detect
    const eventType = replace ? 'replacestate' : 'pushstate';
    window.dispatchEvent(new Event(eventType));
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));

    errorLogger.logError(
      typedError,
      'error',
      {
        component: 'SafeRouting',
        function: 'navigate',
        additionalData: { path, options }
      }
    );

    throw typedError;
  }
}